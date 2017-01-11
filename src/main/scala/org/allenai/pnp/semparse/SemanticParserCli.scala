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
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.ccg.lambda2.ExpressionComparator
import com.jayantkrish.jklol.ccg.lambda2.ExpressionSimplifier
import com.jayantkrish.jklol.ccg.lambda2.SimplificationComparator
import com.jayantkrish.jklol.ccg.lambda2.StaticAnalysis
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
  var entityDataOpt: OptionSpec[String] = null
  var testDataOpt: OptionSpec[String] = null
  
  override def initializeOptions(parser: OptionParser): Unit = {
    trainingDataOpt = parser.accepts("trainingData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',').required()
    entityDataOpt = parser.accepts("entityData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',').required()
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
    
    val entityData = ListBuffer[CcgExample]()
    for (filename <- options.valuesOf(entityDataOpt).asScala) {
      entityData ++= TrainSemanticParser.readCcgExamples(filename).asScala
    }

    val testData = ListBuffer[CcgExample]()
    if (options.has(testDataOpt)) {
      for (filename <- options.valuesOf(testDataOpt).asScala) {
        testData ++= TrainSemanticParser.readCcgExamples(filename).asScala
      }
    }
    
    println(trainingData.size + " training examples")
    println(entityData.size + " entity names")
    println(testData.size + " test examples")
    val wordCounts = getWordCounts(trainingData)
    val entityWordCounts = getWordCounts(entityData)
    
    // Vocab consists of all words that appear more than once in
    // the training data or appear in the entity names.
    val vocab = IndexedList.create(wordCounts.getKeysAboveCountThreshold(1.9))
    vocab.addAll(entityWordCounts.getKeysAboveCountThreshold(0.0))
    vocab.add(SemanticParserCli.UNK)
    
    val entityDict = buildEntityDictionary(entityData, vocab, typeDeclaration)
    
    val trainPreprocessed = trainingData.map(x => preprocessExample(x, simplifier, vocab, entityDict)) 
    val testPreprocessed = testData.map(x => preprocessExample(x, simplifier, vocab, entityDict))

    val actionSpace = SemanticParser.generateActionSpace(
        trainPreprocessed.map(_.getLogicalForm), typeDeclaration)
        
    // Remove entities from the action space, but ensure that there is
    // at least one valid action per type
    for (t <- actionSpace.allTypes) {
      actionSpace.typeTemplateMap.addBinding(t,
          ConstantTemplate(t, Expression2.constant("DUMMY:" + t)))
    }
    for (entity <- entityDict.map.values.flatten) {
      actionSpace.typeTemplateMap.removeBinding(entity.t, entity.template)
    }

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
  
  def buildEntityDictionary(examples: Seq[CcgExample], vocab: IndexedList[String],
      typeDeclaration: TypeDeclaration): EntityDict = {
    val entityNames = ListBuffer[(Expression2, List[Int])]()
    for (ex <- examples) {
      val encodedName = ex.getSentence.getWords.asScala.toList.map(vocab.getIndex(_))
      entityNames += ((ex.getLogicalForm, encodedName))
    }

    val entityNameMap = SemanticParser.seqToMultimap(entityNames)
    val entityDict = ListBuffer[(List[Int], Entity)]()
    for (e <- entityNameMap.keySet) {
      val names = entityNameMap(e).toList
      val t = StaticAnalysis.inferType(e, typeDeclaration)
      val template = ConstantTemplate(t, e)
      val entity = Entity(e, t, template, names)
      for (name <- names) {
        entityDict += ((name, entity))
      }
    }

    new EntityDict(SemanticParser.seqToMultimap(entityDict))
  }

  def preprocessExample(ex: CcgExample, simplifier: ExpressionSimplifier,
      vocab: IndexedList[String], entityDict: EntityDict): CcgExample = {
    val sent = ex.getSentence
    val unkedWords = sent.getWords.asScala.map(
        x => if (vocab.contains(x)) { x } else { SemanticParserCli.UNK })
    val tokenIds = unkedWords.map(x => vocab.getIndex(x)).toList
    val entityLinking = entityDict.link(tokenIds)
    
    val annotations = Maps.newHashMap[String, Object](sent.getAnnotations)
    annotations.put("originalTokens", sent.getWords)
    annotations.put("tokenIds", tokenIds)
    annotations.put("entityLinking", entityLinking)

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

      val sent = e.getSentence
      val tokenIds = sent.getAnnotation("tokenIds").asInstanceOf[List[Int]]
      val entityLinking = sent.getAnnotation("entityLinking").asInstanceOf[EntityLinking]

      val oracle = parser.generateExecutionOracle(e.getLogicalForm, entityLinking, typeDeclaration)
      val dist = parser.parse(tokenIds, entityLinking)

      if (oracle.isDefined) {
        val cg = new ComputationGraph
        val results = dist.beamSearch(1, 50, Env.init, oracle.get,
            model.getInitialComputationGraph(cg), new NullLogFunction())
        if (results.executions.size != 1) {
          println("ERROR: " + e + " " + results)
          println("  " + e.getSentence.getWords)
          println("  " + e.getLogicalForm)
          println("  " + e.getSentence.getAnnotation("entityLinking"))

          numFailed += 1
        } else {
          val numParts = results.executions(0).value.parts.size
          maxParts = Math.max(numParts, maxParts)
          if (results.executions.length > 1) {
            println("MULTIPLE: " + results.executions.length + " " + e)
            println("  " + e.getSentence.getWords)
            println("  " + e.getLogicalForm)
            println("  " + e.getSentence.getAnnotation("entityLinking"))
          } else {
            // println("OK   : " + numParts + " " + " "
          }
        }
        cg.delete
      } else {
        println("ORACLE: " + e)
        println("  " + e.getSentence.getWords)
        println("  " + e.getLogicalForm)
        println("  " + e.getSentence.getAnnotation("entityLinking"))

        numFailed += 1
      }
    }
    println("max parts: " + maxParts)
    println("decoding failures: " + numFailed)
  }
  
  /** Train the parser by maximizing the likelihood of examples.
    * Returns a model with the trained parameters. 
    */
  def train(examples: Seq[CcgExample], parser: SemanticParser,
      typeDeclaration: TypeDeclaration): PpModel = {
    val ppExamples = for {
      x <- examples
      sent = x.getSentence
      tokenIds = sent.getAnnotation("tokenIds").asInstanceOf[List[Int]]
      entityLinking = sent.getAnnotation("entityLinking").asInstanceOf[EntityLinking]
      unconditional = parser.generateExpression(tokenIds, entityLinking)
      oracle <- parser.generateExecutionOracle(x.getLogicalForm, entityLinking, typeDeclaration)
    } yield {
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
      val results = dist.beamSearch(10, 75, Env.init, null,
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