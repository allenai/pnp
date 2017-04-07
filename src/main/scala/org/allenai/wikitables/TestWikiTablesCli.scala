package org.allenai.wikitables

import scala.collection.JavaConverters._

import org.allenai.pnp.Env
import org.allenai.pnp.PnpInferenceContext
import org.allenai.pnp.PnpModel
import org.allenai.pnp.semparse.EntityLinking
import org.allenai.pnp.semparse.SemanticParser
import org.allenai.pnp.semparse.SemanticParserLoss
import org.allenai.pnp.semparse.SemanticParserState

import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration
import com.jayantkrish.jklol.ccg.lambda2.ExpressionComparator
import com.jayantkrish.jklol.ccg.lambda2.ExpressionSimplifier
import com.jayantkrish.jklol.ccg.lambda2.SimplificationComparator
import com.jayantkrish.jklol.cli.AbstractCli

import edu.cmu.dynet._
import joptsimple.OptionParser
import joptsimple.OptionSet
import joptsimple.OptionSpec
import scala.collection.mutable.ListBuffer

class TestWikiTablesCli extends AbstractCli() {

  var testDataOpt: OptionSpec[String] = null
  var derivationsPathOpt: OptionSpec[String] = null
  var modelOpt: OptionSpec[String] = null

  var beamSizeOpt: OptionSpec[Integer] = null
  var evaluateDpdOpt: OptionSpec[Void] = null
  var maxDerivationsOpt: OptionSpec[Integer] = null
  var seq2TreeOpt: OptionSpec[Void] = null

  override def initializeOptions(parser: OptionParser): Unit = {
    testDataOpt = parser.accepts("testData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',').required()
    derivationsPathOpt = parser.accepts("derivationsPath").withRequiredArg().ofType(classOf[String])
    modelOpt = parser.accepts("model").withRequiredArg().ofType(classOf[String]).required()

    beamSizeOpt = parser.accepts("beamSize").withRequiredArg().ofType(classOf[Integer]).defaultsTo(5)
    evaluateDpdOpt = parser.accepts("evaluateDpd")
    maxDerivationsOpt = parser.accepts("maxDerivations").withRequiredArg().ofType(classOf[Integer]).defaultsTo(-1)
    seq2TreeOpt = parser.accepts("seq2Tree")    
  }

  override def run(options: OptionSet): Unit = {
    Initialize.initialize(Map("dynet-mem" -> "2048"))

    // Initialize expression processing for Wikitables logical forms.
    val simplifier = ExpressionSimplifier.lambdaCalculus()
    val comparator = new SimplificationComparator(simplifier)
    val logicalFormParser = ExpressionParser.expression2();
    val entityLinker = new WikiTablesEntityLinker()

    // Read in serialized semantic parser
    val loader = new ModelLoader(options.valueOf(modelOpt))
    val model = PnpModel.load(loader)
    val parser = SemanticParser.load(loader, model)
    val featureGenerator = parser.config.featureGenerator.get
    loader.done()

    val typeDeclaration = if (options.has(seq2TreeOpt)) {
      new Seq2TreeTypeDeclaration()
    } else {
      new WikiTablesTypeDeclaration()
    }
    
    val lfPreprocessor = if (options.has(seq2TreeOpt)) {
      new Seq2TreePreprocessor()
    } else {
      new NullLfPreprocessor()
    }

    // Read test data.
    val testData = WikiTablesUtil.loadDatasets(options.valuesOf(testDataOpt).asScala,
        options.valueOf(derivationsPathOpt), options.valueOf(maxDerivationsOpt),
        lfPreprocessor)
    println("Read " + testData.size + " test examples")

    testData.foreach(x => WikiTablesUtil.preprocessExample(x, parser.vocab,
        featureGenerator, typeDeclaration))

    /*
    println("*** Validating test set action space ***")
    val testSeparatedLfs = WikiTablesSemanticParserCli.getCcgDataset(testPreprocessed)
    SemanticParserUtils.validateActionSpace(testSeparatedLfs, parser, typeDeclaration)
    */

    val testResults = TestWikiTablesCli.test(testData.map(_.ex),
        parser, options.valueOf(beamSizeOpt), options.has(evaluateDpdOpt),
        true, typeDeclaration, comparator, lfPreprocessor, println)
    println("*** Evaluation results ***")
    println(testResults)
  }
}

object TestWikiTablesCli {

  def main(args: Array[String]): Unit = {
    (new TestWikiTablesCli()).run(args)
  }

  /** Evaluate the test accuracy of parser on examples. Logical
   * forms are compared for equality using comparator.
   */
  def test(examples: Seq[WikiTablesExample], parser: SemanticParser, beamSize: Int,
      evaluateDpd: Boolean, evaluateOracle: Boolean, typeDeclaration: TypeDeclaration,
      comparator: ExpressionComparator, preprocessor: LfPreprocessor,
      print: Any => Unit): SemanticParserLoss = {

    print("")
    var numCorrect = 0
    var numCorrectAt10 = 0
    for (e <- examples) {
      val sent = e.sentence
      print("example id: " + e.id +  " " + e.tableString)
      print(sent.getWords.asScala.mkString(" "))
      print(sent.getAnnotation("unkedTokens").asInstanceOf[List[String]].mkString(" "))

      val entityLinking = sent.getAnnotation("entityLinking").asInstanceOf[EntityLinking]
      val dist = parser.parse(sent.getAnnotation("tokenIds").asInstanceOf[Array[Int]],
          entityLinking)

      ComputationGraph.renew()
      val context = PnpInferenceContext.init(parser.model)
      val results = dist.beamSearch(beamSize, 75, Env.init, context)

      val beam = results.executions.slice(0, 10)
      val correct = beam.map { x =>
        val expression = x.value.decodeExpression

        val isCorrect = if (evaluateDpd) {
          // Evaluate the logical forms using the output of dynamic programming on denotations.
          e.logicalForms.size > 0 && e.logicalForms.map(x => comparator.equals(x, expression)).reduce(_ || _)
        } else {
          // Evaluate the logical form by executing it.
          e.isFormulaCorrect(preprocessor.postprocess(expression))
        }
        
        if (isCorrect) {
          print("* " + x.logProb.formatted("%02.3f") + "  " + expression)
          true
        } else {
          print("  " + x.logProb.formatted("%02.3f") + "  " + expression)
          false
        }
      }

      if (correct.length > 0 && correct(0)) {
        numCorrect += 1
      }
      if (correct.fold(false)(_ || _)) {
        numCorrectAt10 += 1
      }

      // Re-parse with a label oracle to find the highest-scoring correct parses.
      if (evaluateOracle) {
        val oracle = parser.getMultiLabelScore(e.logicalForms, entityLinking, typeDeclaration)
        if (oracle.isDefined) { 
          val oracleContext = PnpInferenceContext.init(parser.model).addExecutionScore(oracle.get)
          val oracleResults = dist.beamSearch(beamSize, 75, Env.init, oracleContext)
            
          oracleResults.executions.map { x =>
            val expression = x.value.decodeExpression
            print("o " + x.logProb.formatted("%02.3f") + "  " + expression)
          }
        } else {
          print("  No correct logical forms in oracle.")
        }
      }

      // Print the attentions of the best predicted derivation
      if (beam.nonEmpty) {
        printAttentions(beam(0).value, e.sentence.getWords.asScala.toArray, print)
      }
      
      printEntityTokenFeatures(entityLinking, e.sentence.getWords.asScala.toArray, print)
    }

    val loss = SemanticParserLoss(numCorrect, numCorrectAt10, examples.length)
    loss
  }

  def printAttentions(state: SemanticParserState, tokens: Array[String],
      print: Any => Unit): Unit = {
    val templates = state.getTemplates
    val attentions = state.getAttentions
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

      print("  " + tokenStrings.mkString(" ") + " " + templates(i))
    }
  }
  
  def printEntityTokenFeatures(entityLinking: EntityLinking, tokens: Array[String],
      print: Any => Unit): Unit = {
    for ((entity, features) <- entityLinking.entities.zip(entityLinking.entityTokenFeatures)) {
      val dim = features._1
      val featureMatrix = features._2
      val values = Expression.input(dim, featureMatrix)
      
      for ((token, i) <- tokens.zipWithIndex) {
        val features = ComputationGraph.incrementalForward(Expression.pick(values, i)).toSeq
        if (features.filter(_ != 0.0f).size > 0) {
          print(entity.expr + " " + token + " " + features)
        }
      }
    }
  }
}
