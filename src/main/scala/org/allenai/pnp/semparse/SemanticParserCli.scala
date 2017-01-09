package org.allenai.pnp.semparse

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

import org.allenai.pnp.Env
import org.allenai.pnp.LoglikelihoodTrainer
import org.allenai.pnp.PpExample
import org.allenai.pnp.PpModel

import com.google.common.collect.Maps
import com.jayantkrish.jklol.ccg.CcgExample
import com.jayantkrish.jklol.ccg.cli.TrainSemanticParser
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration
import com.jayantkrish.jklol.ccg.lambda2.ExpressionComparator
import com.jayantkrish.jklol.ccg.lambda2.ExpressionSimplifier
import com.jayantkrish.jklol.ccg.lambda2.SimplificationComparator
import com.jayantkrish.jklol.cli.AbstractCli
import com.jayantkrish.jklol.experiments.geoquery.GeoqueryUtil
import com.jayantkrish.jklol.nlpannotation.AnnotatedSentence
import com.jayantkrish.jklol.training.DefaultLogFunction
import com.jayantkrish.jklol.training.NullLogFunction
import com.jayantkrish.jklol.util.CountAccumulator
import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._
import joptsimple.OptionParser
import joptsimple.OptionSet
import joptsimple.OptionSpec

/** Command line program for training a semantic parser.
  */
class SemanticParserCli extends AbstractCli() {
  
  var trainingDataOpt: OptionSpec[String] = null
  var testDataOpt: OptionSpec[String] = null
  
  override def initializeOptions(parser: OptionParser): Unit = {
    trainingDataOpt = parser.accepts("trainingData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',').required()
    testDataOpt = parser.accepts("testData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',')
  }
  
  override def run(options: OptionSet): Unit = {
    myInitialize()
    
    // Initialize expression processing for Geoquery logical forms. 
    val typeDeclaration = GeoqueryUtil.getTypeDeclaration()
    val simplifier = GeoqueryUtil.getExpressionSimplifier
    val comparator = new SimplificationComparator(simplifier)
    
    // Read and preprocess data
    val trainingData = ListBuffer[CcgExample]()
    for (filename <- options.valuesOf(trainingDataOpt).asScala) {
      trainingData ++= TrainSemanticParser.readCcgExamples(filename).asScala
    }
    
    val testData = ListBuffer[CcgExample]()
    if (options.has(testDataOpt)) {
      for (filename <- options.valuesOf(testDataOpt).asScala) {
        testData ++= TrainSemanticParser.readCcgExamples(filename).asScala
      }
    }
    
    println(trainingData.size + " training examples")
    println(testData.size + " test examples")
    val wordCounts = getWordCounts(trainingData)
    
    // Vocab consists of all words that appear more than once.
    // TODO: entity names... 
    val vocab = IndexedList.create(wordCounts.getKeysAboveCountThreshold(1.9))
    vocab.add(SemanticParserCli.UNK)
    
    val trainPreprocessed = trainingData.map(x => preprocessExample(x, simplifier, vocab)) 
    val testPreprocessed = testData.map(x => preprocessExample(x, simplifier, vocab))
          
    val actionSpace = SemanticParser.generateActionSpace(
        trainPreprocessed.map(_.getLogicalForm), typeDeclaration)
    
    // println(actionSpace.rootTypes)
    // println(actionSpace.typeTemplateMap)
    
    val parser = new SemanticParser(actionSpace, vocab)
    
    validateActionSpace(trainPreprocessed, parser, typeDeclaration)
    val trainedModel = train(trainPreprocessed, parser, typeDeclaration)
    test(testPreprocessed, parser, trainedModel, typeDeclaration, simplifier, comparator)
    test(trainPreprocessed, parser, trainedModel, typeDeclaration, simplifier, comparator)
    
    // TODO: serialization
  }
  
  def getWordCounts(examples: Seq[CcgExample]): CountAccumulator[String] = {
    val acc = CountAccumulator.create[String]
    for (ex <- examples) {
      ex.getSentence.getWords.asScala.map(x => acc.increment(x, 1.0)) 
    }
    acc
  }
  
  def preprocessExample(ex: CcgExample, simplifier: ExpressionSimplifier,
      vocab: IndexedList[String]): CcgExample = {
    val sent = ex.getSentence
    val unkedWords = sent.getWords.asScala.map(
        x => if (vocab.contains(x)) { x } else { SemanticParserCli.UNK })

    val annotations = Maps.newHashMap[String, Object](sent.getAnnotations)
    annotations.put("originalTokens", sent.getWords)
    annotations.put("tokenIds", unkedWords.map(x => vocab.getIndex(x)).toList)

    val unkedSentence = new AnnotatedSentence(unkedWords.asJava,
        sent.getPosTags, annotations)
    
    new CcgExample(unkedSentence, ex.getDependencies, ex.getSyntacticParse, 
          simplifier.apply(ex.getLogicalForm))
  }

  /** Verify that the parser can generate the logical form
    * in each training example when the search is constrained
    * by the execution oracle.  
    */
  def validateActionSpace(examples: Seq[CcgExample], parser: SemanticParser,
      typeDeclaration: TypeDeclaration): Unit = {
    val model = parser.getModel

    println("")
    var maxParts = 0
    var numFailed = 0
    for (e <- examples) {
      println(e.getSentence.getWords)
      println(e.getLogicalForm)

      val oracle = parser.generateExecutionOracle(e.getLogicalForm, typeDeclaration)

      val sent = e.getSentence
      val dist = parser.parse(
          sent.getAnnotation("tokenIds").asInstanceOf[List[Int]],
          sent.getAnnotation("entityLinking").asInstanceOf[EntityLinking])

      val cg = new ComputationGraph
      val results = dist.beamSearch(1, 50, Env.init, oracle,
          model.getInitialComputationGraph(cg), new NullLogFunction())
      if (results.executions.size != 1) {
        println("ERROR: " + e + " " + results)
        numFailed += 1
      } else {
        val numParts = results.executions(0).value.parts.size
        println("OK   : " + numParts + " " + e)
        maxParts = Math.max(numParts, maxParts)
      }
      cg.delete
    }
    println("max parts: " + maxParts)
    println("decoding failures: " + numFailed)
  }
  
  /** Train the parser by maximizing the likelihood of examples.
    * Returns a model with the trained parameters. 
    */
  def train(examples: Seq[CcgExample], parser: SemanticParser,
      typeDeclaration: TypeDeclaration): PpModel = {
    val ppExamples = examples map { x =>
      val sent = x.getSentence
      val unconditional = parser.generateExpression(
          sent.getAnnotation("tokenIds").asInstanceOf[List[Int]],
          sent.getAnnotation("entityLinking").asInstanceOf[EntityLinking])
      val oracle = parser.generateExecutionOracle(x.getLogicalForm, typeDeclaration)
      PpExample(unconditional, unconditional, Env.init, oracle)
    }
    
    // Train model
    val model = parser.getModel
    val sgd = new SimpleSGDTrainer(model.model, 0.1f, 0.01f)
    val trainer = new LoglikelihoodTrainer(100, 100, model, sgd, new DefaultLogFunction())
    trainer.train(ppExamples.toList)

    model
  }

  /** Evaluate the test accuracy of parser on examples. Logical
    * forms are compared for equality using comparator.  
    */
  def test(examples: Seq[CcgExample], parser: SemanticParser,
      model: PpModel, typeDeclaration: TypeDeclaration, simplifier: ExpressionSimplifier,
      comparator: ExpressionComparator): Unit = {
    println("")
    var numCorrect = 0
    for (e <- examples) {
      println(e.getSentence.getWords.asScala.mkString(" "))
      println("  " + e.getLogicalForm)

      val sent = e.getSentence
      val dist = parser.generateExpression(
          sent.getAnnotation("tokenIds").asInstanceOf[List[Int]],
          sent.getAnnotation("entityLinking").asInstanceOf[EntityLinking])
      val cg = new ComputationGraph
      val results = dist.beamSearch(100, 75, Env.init, null,
          model.getInitialComputationGraph(cg), new NullLogFunction())
          
      val bestLf = simplifier.apply(results.executions(0).value)
      if (comparator.equals(e.getLogicalForm, bestLf)) {
        numCorrect += 1
        println("C " + bestLf)
      } else {
        println("I " + bestLf)
      }
      
      println()
      cg.delete
    }
    
    val accuracy = numCorrect.asInstanceOf[Double] / examples.length 
    println(accuracy + " " + numCorrect + " / " + examples.length)
  }
}

object SemanticParserCli {
  
  val UNK = "<UNK>"
  
  def main(args: Array[String]): Unit = {
    (new SemanticParserCli()).run(args)
  }
}