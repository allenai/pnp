package org.allenai.wikitables

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer
import java.util.HashSet

import org.allenai.pnp.Env
import org.allenai.pnp.LoglikelihoodTrainer
import org.allenai.pnp.PnpExample
import org.allenai.pnp.PnpModel
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
import edu.stanford.nlp.sempre.tables.test.CustomExample
import joptsimple.OptionParser
import joptsimple.OptionSet
import joptsimple.OptionSpec
import org.allenai.pnp.semparse.ActionSpace
import org.allenai.pnp.PnpInferenceContext
import org.allenai.pnp.semparse.SemanticParserConfig

/** Command line program for training a semantic parser 
  * on the WikiTables data set.
  * runMain org.allenai.wikitables.WikiTablesSemanticParserCli -trainingData TRAIN-DATA-PATH
  *                                                           [-testData TEST-DATA-PATH]
  *                                                           [-derivationsPath PATH-TO-LOGICAL-FORMS]
  * If derivationsPath is not specified, Sempre will be used to parse utterances (this will be SLOW!)
  */
class WikiTablesSemanticParserCli extends AbstractCli() {
  
  var trainingDataOpt: OptionSpec[String] = null
  // Path to the directory containing the correct logical forms
  var derivationsPathOpt: OptionSpec[String] = null
  var testDataOpt: OptionSpec[String] = null
  
  var skipActionSpaceValidationOpt: OptionSpec[Void] = null
  var trainOnAnnotatedLfsOpt: OptionSpec[Void] = null

  override def initializeOptions(parser: OptionParser): Unit = {
    trainingDataOpt = parser.accepts("trainingData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',').required()
    derivationsPathOpt = parser.accepts("derivationsPath").withRequiredArg().ofType(classOf[String])
    testDataOpt = parser.accepts("testData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',')

    skipActionSpaceValidationOpt = parser.accepts("skipActionSpaceValidation")
    trainOnAnnotatedLfsOpt = parser.accepts("trainOnAnnotatedLfs")
  }
  
  override def run(options: OptionSet): Unit = {
    Initialize.initialize(Map("dynet-mem" -> "2048"))

    // Initialize expression processing for Wikitables logical forms. 
    val simplifier = ExpressionSimplifier.lambdaCalculus()
    val comparator = new SimplificationComparator(simplifier)
    val logicalFormParser = ExpressionParser.expression2();
    val typeDeclaration = new WikiTablesTypeDeclaration()

    // Read and preprocess data
    val trainingData = ListBuffer[CustomExample]()
    val includeDerivationsForTrain = !options.has(trainOnAnnotatedLfsOpt)
    for (filename <- options.valuesOf(trainingDataOpt).asScala) {
      trainingData ++= WikiTablesDataProcessor.getDataset(filename, true, includeDerivationsForTrain,
        options.valueOf(derivationsPathOpt), 100, 50).asScala
    }
    
    val testData = ListBuffer[CustomExample]()
    if (options.has(testDataOpt)) {
      for (filename <- options.valuesOf(testDataOpt).asScala) {
        testData ++= WikiTablesDataProcessor.getDataset(filename, true, true,
          options.valueOf(derivationsPathOpt), 100, -1).asScala
      }
    }
    
    println("Read " + trainingData.size + " training examples")
    val wordCounts = getWordCounts(trainingData)
    val allEntities = trainingData.map(ex => getUnlinkedEntities(ex)).flatten.toList
    val entityCounts = getEntityCounts(allEntities)
    // Vocab consists of all words that appear more than once in
    // the training data.
    val vocab = IndexedList.create(wordCounts.getKeysAboveCountThreshold(1.9))
    vocab.addAll(IndexedList.create(entityCounts.getKeysAboveCountThreshold(0.0)))
    vocab.add(WikiTablesSemanticParserCli.UNK)
    vocab.add(WikiTablesSemanticParserCli.ENTITY)
    println(vocab.size + " words")
    println("Read " + testData.size + " test examples")

    // Eliminate those examples that Sempre did not find correct logical forms for.
    val trainPreprocessed = trainingData.filter(!_.alternativeFormulas.isEmpty).map(
        x => preprocessExample(x, vocab, simplifier, logicalFormParser, typeDeclaration))
    val testPreprocessed = testData.filter(!_.alternativeFormulas.isEmpty).map(
        x => preprocessExample(x, vocab, simplifier, logicalFormParser, typeDeclaration))

    println("Found correct logical forms for " + trainPreprocessed.size + " training examples")
    println("Found correct logical forms for " + testPreprocessed.size + " test examples")

    // Generate the action space of the semantic parser from the logical
    // forms that are well-typed.
    val trainingLfs = trainPreprocessed.map(_.getLogicalForms.asScala).flatten
    println("*** Validating types ***")
    val wellTypedTrainingLfs = trainingLfs.filter(lf =>
      SemanticParserUtils.validateTypes(lf, typeDeclaration))
      
    println("*** Generating action space ***")
    val actionSpace = ActionSpace.fromExpressions(wellTypedTrainingLfs, typeDeclaration, false)

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

    val model = PnpModel.init(true)
    val config = new SemanticParserConfig()
    config.attentionCopyEntities = false
    val parser = SemanticParser.create(actionSpace, vocab, config, model)
    
    if (!options.has(skipActionSpaceValidationOpt)) {
      val trainSeparatedLfs = getCcgDataset(trainPreprocessed)
      val testSeparatedLfs = getCcgDataset(testPreprocessed)

      println("*** Validating train set action space ***")
      SemanticParserUtils.validateActionSpace(trainSeparatedLfs, parser, typeDeclaration)
      println("*** Validating test set action space ***")
      SemanticParserUtils.validateActionSpace(testSeparatedLfs, parser, typeDeclaration)
    } else {
      println("Skipping action space validation")
    }

    val trainedModel = train(trainPreprocessed, parser, typeDeclaration)
    println("***************** TEST EVALUATION *****************")
    val testResults = test(testPreprocessed, parser, trainedModel, typeDeclaration, comparator)
    println("***************** TRAIN EVALUATION *****************")
    val trainResults = test(trainPreprocessed, parser, trainedModel, typeDeclaration, comparator)
    
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

  def getEntityCounts(entityNames: List[String]): CountAccumulator[String] = {
    val acc = CountAccumulator.create[String]
    for (entityString <- entityNames) {
      // Strings are of the form fb:cell.cell_name
      val parts = entityString.split('.')
      acc.increment(parts.slice(1, parts.size - 1).mkString("."), 1.0)  // entity type
      parts.last.split('_').map(x => acc.increment(x, 1.0))  // actual value split into tokens
    }
    acc
  }

  def getUnlinkedEntities(ex: CustomExample): List[String] = {
    val sempreEntityLinking = WikiTablesDataProcessor.getEntityLinking(ex)
    sempreEntityLinking.asScala.filter(p => p.getFirst() == null).map(p => p.getSecond().toString).toList
  }
  
  /**
   * Converts a {@code CustomExample} into a {@code WikiTablesExample}. 
   */
  def preprocessExample(ex: CustomExample, vocab: IndexedList[String],
                        simplifier: ExpressionSimplifier,
                        lfParser: ExpressionParser[Expression2],
                        typeDeclaration: WikiTablesTypeDeclaration): WikiTablesExample = {
    val sent = new AnnotatedSentence(ex.getTokens(), ex.languageInfo.posTags)
    val unkedWords = sent.getWords.asScala.map(
        x => if (vocab.contains(x)) { x } else { WikiTablesSemanticParserCli.UNK })
    val tokenIds = unkedWords.map(x => vocab.getIndex(x)).toList
    val sempreEntityLinking = WikiTablesDataProcessor.getEntityLinking(ex)
    val builder = ListBuffer[(Option[Span], Entity, List[Int], Double)]()
    for (linking <- sempreEntityLinking.asScala) {
      val entityString = linking.getSecond.toString
      val entityExpr = Expression2.constant(entityString)
      val entityType = typeDeclaration.getType(entityString)
      val template = ConstantTemplate(entityType, entityExpr)
      // Note: Passing a constant score of 0.1 for all matches
      if (linking.getFirst() == null) {
        val entityTokens = tokenizeEntity(entityString)
        val entityTokenIds = entityTokens.map(x => 
          if (vocab.contains(x)) {
            vocab.getIndex(x)
          } else {
            vocab.getIndex(WikiTablesSemanticParserCli.UNK)
          } ).toList
        val entity = Entity(entityExpr, entityType, template, List(entityTokenIds))
        builder += ((None, entity, entityTokenIds, 0.1))
      } else {
        val i = linking.getFirst().getFirst()
        val j = linking.getFirst().getSecond()
        // val entityTokenIds = tokenIds.slice(i, j)

        // XXX: The vocabulary isn't constructed to contain the tokens
        // found in entities. This should be fixed.
        val entityTokens = tokenizeEntity(entityString)
        val entityTokenIds = entityTokens.map(x => 
          if (vocab.contains(x)) {
            vocab.getIndex(x)
          } else {
            vocab.getIndex(WikiTablesSemanticParserCli.UNK)
          } ).toList

        val entity = Entity(entityExpr, entityType, template, List(entityTokenIds))
        builder += ((Some(Span(i, j)), entity, entityTokenIds, 0.1))
      }
    }
    val entityLinking = new EntityLinking(builder.toList)

    val entityAnonymizedWords = unkedWords.toArray
    val entityAnonymizedTokenIds = tokenIds.toArray
    // Don't anonymize entity words.
    /*
    for (entityMatch <- entityLinking.linkedMatches) {
      val span = entityMatch._1
      for (i <- span.start until span.end) {
        entityAnonymizedTokenIds(i) = vocab.getIndex(WikiTablesSemanticParserCli.ENTITY)
        entityAnonymizedWords(i) = WikiTablesSemanticParserCli.ENTITY
      }
    }
    */

    val annotations = Maps.newHashMap[String, Object](sent.getAnnotations)
    annotations.put("originalTokens", sent.getWords.asScala.toList)
    annotations.put("tokenIds", entityAnonymizedTokenIds.toArray)
    annotations.put("entityLinking", entityLinking)

    val unkedSentence = new AnnotatedSentence(entityAnonymizedWords.toList.asJava,
        sent.getPosTags, annotations)

    val correctLogicalForms = {
      if (ex.targetFormula == null) {
        ex.alternativeFormulas.asScala.map { x => WikiTablesUtil.toPnpLogicalForm(x) }
      } else {
        // This means we have the gold annotation available.
        Seq(WikiTablesUtil.toPnpLogicalForm(ex.targetFormula))
      }
    }
    val parsedLogicalForms = correctLogicalForms.map {x => simplifier.apply(lfParser.parse(x))}
    new WikiTablesExample(unkedSentence, new HashSet[Expression2](parsedLogicalForms.asJava),
                          ex.context, ex.targetValue);
  }
  
  /**
   * Converts the id of an entity to a sequence of
   * tokens in its "name." The tokens will have the form
   * [type, token1, token2, ..]. For example:
   * "fb:row.row.player_name" -> ["fb:row.row", "player", "name"]
   */
  def tokenizeEntity(entityString: String): List[String] = {
    val lastDotInd = entityString.lastIndexOf(".")
    val entityTokens = if (lastDotInd == -1) {
      entityString.split('_').toList
    } else {
      val (typeName, entityName) = entityString.splitAt(lastDotInd)
      List(typeName) ++ entityName.substring(1).split('_')
    }

    // println("tokens: " + entityString + " " + entityTokens)
    entityTokens
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
      typeDeclaration: TypeDeclaration): PnpModel = {
    
    parser.dropoutProb = 0.5
    val pnpExamples = for {
      x <- examples
      sent = x.getSentence
      tokenIds = sent.getAnnotation("tokenIds").asInstanceOf[Array[Int]]
      entityLinking = sent.getAnnotation("entityLinking").asInstanceOf[EntityLinking]
      unconditional = parser.generateExpression(tokenIds, entityLinking)
      oracle <- parser.getMultiLabelScore(x.getLogicalForms.asScala, entityLinking, typeDeclaration)
    } yield {
      PnpExample(unconditional, unconditional, Env.init, oracle)
    }

    println(pnpExamples.size + " training examples after oracle generation.")

    // Train model
    val model = parser.model
    val sgd = new SimpleSGDTrainer(model.model, 0.1f, 0.01f)
    val trainer = new LoglikelihoodTrainer(20, 5, true, model, sgd, new DefaultLogFunction())
    trainer.train(pnpExamples.toList)

    parser.dropoutProb = -1
    model
  }

  //TODO(pradeep): Make a new Test Cli
  /** Evaluate the test accuracy of parser on examples. Logical
    * forms are compared for equality using comparator.  
    */
  def test(examples: Seq[WikiTablesExample], parser: SemanticParser,
      model: PnpModel, typeDeclaration: TypeDeclaration, comparator: ExpressionComparator): SemanticParserLoss = {
    println("")
    var numCorrect = 0
    var numCorrectAt10 = 0
    for (e <- examples) {
      val sent = e.getSentence
      println(sent.getWords.asScala.mkString(" "))
      println(sent.getAnnotation("originalTokens").asInstanceOf[List[String]].mkString(" "))

      val entityLinking = sent.getAnnotation("entityLinking").asInstanceOf[EntityLinking]
      val dist = parser.parse(sent.getAnnotation("tokenIds").asInstanceOf[Array[Int]],
          entityLinking)
      
      ComputationGraph.renew()
      val context = PnpInferenceContext.init(model)
      val results = dist.beamSearch(10, 75, Env.init, context)
      
      val beam = results.executions.slice(0, 10)
      val correct = beam.map { x =>
        val expression = x.value.decodeExpression
        if (e.isFormulaCorrect(expression)) {
          println("* " + x.logProb.formatted("%02.3f") + "  " + expression)
          true
        } else {
          println("  " + x.logProb.formatted("%02.3f") + "  " + expression)
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
      if (beam.nonEmpty) {
        val state = beam(0).value
        val templates = state.getTemplates
        val attentions = state.getAttentions
        val tokens = e.getSentence.getWords.asScala.toArray
        for (i <- 0 until templates.length) {
          val values = ComputationGraph.incrementalForward(attentions(i)).toSeq()
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
      }
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
