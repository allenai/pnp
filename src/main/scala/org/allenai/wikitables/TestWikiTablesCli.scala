package org.allenai.wikitables

import scala.collection.JavaConverters._
import com.jayantkrish.jklol.cli.AbstractCli
import joptsimple.OptionParser
import edu.cmu.dynet.Initialize
import joptsimple.OptionSpec
import joptsimple.OptionSet
import scala.collection.mutable.ListBuffer
import edu.stanford.nlp.sempre.tables.test.CustomExample
import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import com.jayantkrish.jklol.ccg.lambda2.SimplificationComparator
import com.jayantkrish.jklol.ccg.lambda2.ExpressionSimplifier
import com.jayantkrish.jklol.ccg.lambda2.ExpressionComparator
import org.allenai.pnp.semparse.EntityLinking
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration
import org.allenai.pnp.semparse.SemanticParser
import org.allenai.pnp.PnpModel
import org.allenai.pnp.PnpInferenceContext
import org.allenai.pnp.semparse.SemanticParserLoss
import edu.cmu.dynet.ComputationGraph
import org.allenai.pnp.Env
import edu.cmu.dynet.ModelLoader

class TestWikiTablesCli extends AbstractCli() {
  
  var testDataOpt: OptionSpec[String] = null
  var derivationsPathOpt: OptionSpec[String] = null
  
  var modelOpt: OptionSpec[String] = null

  override def initializeOptions(parser: OptionParser): Unit = {
    testDataOpt = parser.accepts("testData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',').required()
    derivationsPathOpt = parser.accepts("derivationsPath").withRequiredArg().ofType(classOf[String])
    
    modelOpt = parser.accepts("model").withRequiredArg().ofType(classOf[String]).required()
  }
  
  override def run(options: OptionSet): Unit = {
    Initialize.initialize(Map("dynet-mem" -> "2048"))
    
    // Initialize expression processing for Wikitables logical forms. 
    val simplifier = ExpressionSimplifier.lambdaCalculus()
    val comparator = new SimplificationComparator(simplifier)
    val logicalFormParser = ExpressionParser.expression2();
    val typeDeclaration = new WikiTablesTypeDeclaration()

    // Read test data.
    val testData = ListBuffer[CustomExample]()
    if (options.has(testDataOpt)) {
      for (filename <- options.valuesOf(testDataOpt).asScala) {
        testData ++= WikiTablesDataProcessor.getDataset(filename, true, true, options.valueOf(derivationsPathOpt), 100, -1).asScala
      }
    }
    
    // Read in serialized semantic parser
    val loader = new ModelLoader(options.valueOf(modelOpt))
    val model = PnpModel.load(loader)
    val parser = SemanticParser.load(loader, model)
    loader.done()

    println("Read " + testData.size + " test examples")
    val testPreprocessed = testData.filter(!_.alternativeFormulas.isEmpty).map(
        x => WikiTablesSemanticParserCli.preprocessExample(x, parser.vocab, simplifier, logicalFormParser, typeDeclaration))
        
    val testResults = test(testPreprocessed, parser, model, typeDeclaration, comparator)
    println("*** Evaluation results ***")
    println(testResults)
  }

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
    loss
  }
}

object TestWikiTablesCli {

  def main(args: Array[String]): Unit = {
    (new TestWikiTablesCli()).run(args)
  }
}
