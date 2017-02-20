package org.allenai.wikitables

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

import java.util.HashSet

import org.allenai.pnp.Env
import org.allenai.pnp.LoglikelihoodTrainer
import org.allenai.pnp.PpExample
import org.allenai.pnp.PpModel
import org.allenai.pnp.ExecutionScore
import org.allenai.pnp.semparse.MaxExecutionScore
import org.allenai.pnp.semparse.ConstantTemplate
import org.allenai.pnp.semparse.Entity
import org.allenai.pnp.semparse.EntityLinking
import org.allenai.pnp.semparse.SemanticParser
import org.allenai.pnp.semparse.SemanticParserLoss
import org.allenai.pnp.semparse.SemanticParserUtils
import org.allenai.pnp.semparse.Span

import com.google.common.collect.Maps
import com.jayantkrish.jklol.ccg.CcgExample
import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import com.jayantkrish.jklol.ccg.lambda.Type
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.ccg.lambda2.ExpressionComparator
import com.jayantkrish.jklol.ccg.lambda2.ExpressionSimplifier
import com.jayantkrish.jklol.ccg.lambda2.SimplificationComparator
import com.jayantkrish.jklol.cli.AbstractCli
import com.jayantkrish.jklol.nlpannotation.AnnotatedSentence
import com.jayantkrish.jklol.training.DefaultLogFunction
import com.jayantkrish.jklol.training.NullLogFunction
import com.jayantkrish.jklol.util.CountAccumulator
import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._
import edu.stanford.nlp.sempre.tables.test.CustomExample
import joptsimple.OptionParser
import joptsimple.OptionSet
import joptsimple.OptionSpec
import org.allenai.pnp.semparse.ActionSpace

/** Command line program for training a semantic parser.
  */
class WikiTablesSemanticParserCli extends AbstractCli() {
  
  var trainingDataOpt: OptionSpec[String] = null
  var testDataOpt: OptionSpec[String] = null
  
  override def initializeOptions(parser: OptionParser): Unit = {
    trainingDataOpt = parser.accepts("trainingData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',').required()
    testDataOpt = parser.accepts("testData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',')
  }
  
  override def run(options: OptionSet): Unit = {
    initialize(new DynetParams())
    
    // Initialize expression processing for Wikitables logical forms. 
    val typeDeclaration = new WikiTablesTypeDeclaration()
    val simplifier = ExpressionSimplifier.lambdaCalculus()
    val comparator = new SimplificationComparator(simplifier)
    
    // Read and preprocess data
    val trainingData = ListBuffer[CustomExample]()
    for (filename <- options.valuesOf(trainingDataOpt).asScala) {
      trainingData ++= WikiTablesDataProcessor.getDataset(filename, true, true, 1000).asScala
    }
    
    val testData = ListBuffer[CustomExample]()
    if (options.has(testDataOpt)) {
      for (filename <- options.valuesOf(testDataOpt).asScala) {
        testData ++= WikiTablesDataProcessor.getDataset(filename, true, true, 1000).asScala
      }
    }
    
    println("Read " + trainingData.size + " training examples")
    val wordCounts = getWordCounts(trainingData)
    // Vocab consists of all words that appear more than once in
    // the training data.
    val vocab = IndexedList.create(wordCounts.getKeysAboveCountThreshold(1.9))
    vocab.add(WikiTablesSemanticParserCli.UNK)
    vocab.add(WikiTablesSemanticParserCli.ENTITY)
    println(vocab.size + " words")
    println("Read " + testData.size + " test examples")

    val logicalFormParser = ExpressionParser.expression2();
    // Eliminate those examples that Sempre did not find correct logical forms for.
    val trainPreprocessed = trainingData.flatMap(x => if (x.alternativeFormulas.isEmpty) None else Some(preprocessExample(x, vocab, logicalFormParser)))
    val testPreprocessed = testData.flatMap(x => if (x.alternativeFormulas.isEmpty) None else Some(preprocessExample(x, vocab, logicalFormParser)))
    
    println("Found correct logical forms for " + trainPreprocessed.size + " training examples")
    println("Found correct logical forms for " + testPreprocessed.size + " test examples")

    val actionSpace = ActionSpace.fromExpressions(
        trainPreprocessed.map(_.getLogicalForms.asScala).flatten, typeDeclaration, false)


    // Remove specific numbers/rows/cells from the action space.
    // These need to be added back in on a per-table basis.
    val filterTypes = Seq(Type.parseFrom("i"), Type.parseFrom("c"), Type.parseFrom("p"), Type.parseFrom("<c,r>"))
    for (t <- filterTypes) {
      val templates = actionSpace.typeTemplateMap.getOrElse(t, List()).toSet
      for (template <- templates) {
        if (template.isInstanceOf[ConstantTemplate]) {
          actionSpace.typeTemplateMap.removeBinding(t, template)
        }
      }

      // Create a dummy action for the type to ensure that at least
      // one valid action is always possible.
      if (actionSpace.typeTemplateMap.getOrElse(t, Set()).size == 0) {
        actionSpace.typeTemplateMap.addBinding(t,
            ConstantTemplate(t, Expression2.constant("DUMMY:" + t)))
      }
    }

    // Print out the action space
    for (t <- actionSpace.typeTemplateMap.keys) {
      println(t)
      for (template <- actionSpace.typeTemplateMap(t)) {
        println("  " + template)
      }
    }

    val model = PpModel.init(true)
    val parser = SemanticParser.create(actionSpace, vocab, model)
    
    val trainSeparatedLfs = getCcgDataset(trainPreprocessed)
    val testSeparatedLfs = getCcgDataset(testPreprocessed)
    

    println("*** Validating types ***")
    SemanticParserUtils.validateTypes(trainSeparatedLfs, typeDeclaration)
    println("*** Validating train set action space ***")
    SemanticParserUtils.validateActionSpace(trainSeparatedLfs, parser, typeDeclaration)
    println("*** Validating test set action space ***")
    SemanticParserUtils.validateActionSpace(testSeparatedLfs, parser, typeDeclaration)

    val trainedModel = train(trainPreprocessed, parser, typeDeclaration)
    println("***************** TEST EVALUATION *****************")
    val testResults = test(testPreprocessed, parser, trainedModel, typeDeclaration, simplifier, comparator)
    println("***************** TRAIN EVALUATION *****************")
    val trainResults = test(trainPreprocessed, parser, trainedModel, typeDeclaration, simplifier, comparator)
    
    println("Test: ")
    println(testResults)
    println("Train: ")
    println(trainResults)
    // TODO: serialization
  }
  
  def getWordCounts(examples: Seq[CustomExample]): CountAccumulator[String] = {
    val acc = CountAccumulator.create[String]
    for (ex <- examples) {
      ex.getTokens.asScala.map(x => acc.increment(x, 1.0)) 
    }
    acc
  }
  
  def preprocessExample(ex: CustomExample, vocab: IndexedList[String],
                        lfParser: ExpressionParser[Expression2]): WikiTablesExample = {
    // Takes a CustomExample and returns a WikiTablesExample
    val sent = new AnnotatedSentence(ex.getTokens(), ex.languageInfo.posTags)
    val unkedWords = sent.getWords.asScala.map(
        x => if (vocab.contains(x)) { x } else { WikiTablesSemanticParserCli.UNK })
    val tokenIds = unkedWords.map(x => vocab.getIndex(x)).toList
    val sempreEntityLinking = WikiTablesDataProcessor.getEntityLinking(ex)
    val builder = ListBuffer[(Span, Entity, List[Int], Double)]()
    for (linking <- sempreEntityLinking.asScala) {
      val entityExpr = Expression2.constant(linking.getSecond().toString)
      val entityType = WikiTablesTypeDeclaration.CELL_TYPE
      val template = ConstantTemplate(entityType, entityExpr)
      // TODO: Passing names as null. Not sure what happens.
      val entity = Entity(entityExpr, entityType, template, null)
      // Note: Passing a constant score of 0.1 for all matches
      builder += ((Span(linking.getFirst().getFirst(), linking.getFirst().getSecond()), entity, tokenIds, 0.1))
    }
    val entityLinking = new EntityLinking(builder.toList)
    
    val entityAnonymizedWords = unkedWords.toArray
    val entityAnonymizedTokenIds = tokenIds.toArray
    for (entityMatch <- entityLinking.matches) {
      val span = entityMatch._1
      for (i <- span.start until span.end) {
        entityAnonymizedTokenIds(i) = vocab.getIndex(WikiTablesSemanticParserCli.ENTITY)
        entityAnonymizedWords(i) = WikiTablesSemanticParserCli.ENTITY
      }
    }

    val annotations = Maps.newHashMap[String, Object](sent.getAnnotations)
    annotations.put("originalTokens", sent.getWords.asScala.toList)
    annotations.put("tokenIds", entityAnonymizedTokenIds.toArray)
    annotations.put("entityLinking", entityLinking)

    val unkedSentence = new AnnotatedSentence(entityAnonymizedWords.toList.asJava,
        sent.getPosTags, annotations)
   
    // Sempre's logical forms do not have parens around x in lambda expressions. Fixing that.
    // TODO: This is fragile.
    val correctLogicalForms = ex.alternativeFormulas.asScala.map {x => x.toString().replaceAll("lambda x", "lambda (x)")}
    val parsedLogicalForms = correctLogicalForms.map {x => lfParser.parse(x)}
    new WikiTablesExample(unkedSentence, new HashSet[Expression2](parsedLogicalForms.asJava))
  }

  /** Returns a modified dataset that has CcgExamples, with one logical form per example.
    * Useful for reusing existing methods for validating types and action spaces, etc.
    */
  def getCcgDataset(dataset: Seq[WikiTablesExample]): Seq[CcgExample] = {
    val ccgDataset = for {
      ex <- dataset
      lf <- ex.getLogicalForms.asScala
      ccgExample = new CcgExample(ex.getSentence, null, null, lf)
    } yield {
      ccgExample
    }
    ccgDataset
  }

  /** Train the parser by maximizing the likelihood of examples.
    * Returns a model with the trained parameters. 
    */
  def train(examples: Seq[WikiTablesExample], parser: SemanticParser,
      typeDeclaration: TypeDeclaration): PpModel = {
    
    parser.dropoutProb = 0.5
    val ppExamples = for {
      x <- examples
      sent = x.getSentence
      tokenIds = sent.getAnnotation("tokenIds").asInstanceOf[Array[Int]]
      entityLinking = sent.getAnnotation("entityLinking").asInstanceOf[EntityLinking]
      unconditional = parser.generateExpression(tokenIds, entityLinking)
      oracles = for {
        lf <- x.getLogicalForms.asScala
        oracle <- parser.generateExecutionOracle(lf, entityLinking, typeDeclaration)
      } yield {
        oracle
      }
      oracle = new MaxExecutionScore(oracles.toSeq)
    } yield {
      PpExample(unconditional, unconditional, Env.init, oracle)
    }

    // Train model
    val model = parser.model
    val sgd = new SimpleSGDTrainer(model.model, 0.1f, 0.01f)
    val trainer = new LoglikelihoodTrainer(50, 100, true, model, sgd, new DefaultLogFunction())
    trainer.train(ppExamples.toList)

    parser.dropoutProb = -1
    model
  }

  /** Evaluate the test accuracy of parser on examples. Logical
    * forms are compared for equality using comparator.  
    */
  def test(examples: Seq[WikiTablesExample], parser: SemanticParser,
      model: PpModel, typeDeclaration: TypeDeclaration, simplifier: ExpressionSimplifier,
      comparator: ExpressionComparator): SemanticParserLoss = {
    // TODO: Change the errors to denotation errors.
    println("")
    var numCorrect = 0
    var numCorrectAt10 = 0
    for (e <- examples) {
      println(e.getSentence.getWords.asScala.mkString(" "))
      println(e.getSentence.getAnnotation("originalTokens").asInstanceOf[List[String]].mkString(" "))
      println("expected: " + e.getLogicalForm)
      
      val sent = e.getSentence
      val dist = parser.parse(
          sent.getAnnotation("tokenIds").asInstanceOf[Array[Int]],
          sent.getAnnotation("entityLinking").asInstanceOf[EntityLinking])
      val cg = new ComputationGraph
      val results = dist.beamSearch(10, 75, Env.init, null,
          model.getComputationGraph(cg), new NullLogFunction())
          
      val beam = results.executions.slice(0, 10)
      val correct = beam.map { x =>
        val simplified = simplifier.apply(x.value.decodeExpression)
        if (comparator.equals(e.getLogicalForm, simplified)) {
          println("* " + x.logProb.formatted("%02.3f") + "  " + simplified)
          true
        } else {
          println("  " + x.logProb.formatted("%02.3f") + "  " + simplified)
          false
        }
      }
      
      if (correct.length > 0 && correct(0)) {
        numCorrect += 1
      }
      if (correct.fold(false)(_ || _)) {
        numCorrectAt10 += 1
      }
      
      // Print the attentions of the best predicted derivation
      val state = beam(0).value
      val templates = state.getTemplates
      val attentions = state.getAttentions
      val tokens = e.getSentence.getWords.asScala.toArray
      for (i <- 0 until templates.length) {
        val floatVector = as_vector(cg.get_value(attentions(i)))
        val values = for {
          j <- 0 until floatVector.size().asInstanceOf[Int]
        } yield {
          floatVector.get(j)
        }
        
        val maxIndex = values.zipWithIndex.max._2
        
        val tokenStrings = for {
          j <- 0 until values.length
        } yield {
          val color = if (j == maxIndex) {
            Console.RED
          } else if (values(j) > 0.1) {
            Console.YELLOW
          } else {
            Console.RESET
          }
          
          color + tokens(j) + Console.RESET
        }

        println("  " + tokenStrings.mkString(" ") + " " + templates(i))
      }
      
      cg.delete
    }
    
    val loss = SemanticParserLoss(numCorrect, numCorrectAt10, examples.length)
    println(loss)
    loss
  }
}

object WikiTablesSemanticParserCli {
  
  val UNK = "<UNK>"
  val ENTITY = "<ENTITY>"

  def main(args: Array[String]): Unit = {
    (new WikiTablesSemanticParserCli()).run(args)
  }
}

/**
case class SemanticParserLoss(numCorrect: Int, oracleNumCorrect: Int, numExamples: Int) {
  val accuracy: Double = numCorrect.asInstanceOf[Double] / numExamples
  val oracleAccuracy: Double = oracleNumCorrect.asInstanceOf[Double] / numExamples
  
  override def toString(): String = {
    "accuracy: " + accuracy + " " + numCorrect + " / " + numExamples + "\n" +
    "oracle  : " + oracleAccuracy + " " + oracleNumCorrect + " / " + numExamples  
  }
}
*/
