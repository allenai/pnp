package org.allenai.pnp.semparse

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

import org.allenai.pnp.Env
import org.allenai.pnp.PpModel

import com.jayantkrish.jklol.ccg.CcgExample
import com.jayantkrish.jklol.ccg.cli.TrainSemanticParser
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration
import com.jayantkrish.jklol.ccg.lambda2.ExpressionComparator
import com.jayantkrish.jklol.ccg.lambda2.ExpressionSimplifier
import com.jayantkrish.jklol.ccg.lambda2.SimplificationComparator
import com.jayantkrish.jklol.cli.AbstractCli
import com.jayantkrish.jklol.experiments.geoquery.GeoqueryUtil
import com.jayantkrish.jklol.training.NullLogFunction

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._
import joptsimple.OptionParser
import joptsimple.OptionSet
import joptsimple.OptionSpec


class TestSemanticParserCli extends AbstractCli() {
  
  var entityDataOpt: OptionSpec[String] = null
  var testDataOpt: OptionSpec[String] = null
  var modelOpt: OptionSpec[String] = null
  
  override def initializeOptions(parser: OptionParser): Unit = {
    entityDataOpt = parser.accepts("entityData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',').required()
    testDataOpt = parser.accepts("testData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',').required()
    modelOpt = parser.accepts("model").withRequiredArg().ofType(classOf[String]).required()
  }

  override def run(options: OptionSet): Unit = {
    initialize(SemanticParserUtils.DYNET_PARAMS)
    
    // Initialize expression processing for Geoquery logical forms. 
    val typeDeclaration = GeoqueryUtil.getSimpleTypeDeclaration()
    val simplifier = GeoqueryUtil.getExpressionSimplifier
    val comparator = new SimplificationComparator(simplifier)
    
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
    println(testData.size + " test examples")

    val loader = new ModelLoader(options.valueOf(modelOpt))
    val model = PpModel.load(loader)
    val parser = SemanticParser.load(loader, model)
    loader.done()

    val vocab = parser.vocab

    val entityDict = TrainSemanticParserCli.buildEntityDictionary(entityData,
        vocab, typeDeclaration)
    
    val testPreprocessed = testData.map(x =>
      TrainSemanticParserCli.preprocessExample(x, simplifier, vocab, entityDict))

    println("*** Running Evaluation ***")
    val results = test(testPreprocessed, parser, typeDeclaration, simplifier, comparator)
  }
  
  /** Evaluate the test accuracy of parser on examples. Logical
   *  forms are compared for equality using comparator.  
   */
  def test(examples: Seq[CcgExample], parser: SemanticParser,
      typeDeclaration: TypeDeclaration, simplifier: ExpressionSimplifier,
      comparator: ExpressionComparator): SemanticParserLoss = {
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
      val cg = ComputationGraph.getNew
      val results = dist.beamSearch(5, 75, Env.init, null,
          parser.model.getComputationGraph(cg), new NullLogFunction())
          
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
      if (beam.length > 0) {
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
      }
    }
    
    val loss = SemanticParserLoss(numCorrect, numCorrectAt10, examples.length)
    println(loss)
    loss
  }
}

case class SemanticParserLoss(numCorrect: Int, oracleNumCorrect: Int, numExamples: Int) {
  val accuracy: Double = numCorrect.asInstanceOf[Double] / numExamples
  val oracleAccuracy: Double = oracleNumCorrect.asInstanceOf[Double] / numExamples
  
  override def toString(): String = {
    "accuracy: " + accuracy + " " + numCorrect + " / " + numExamples + "\n" +
    "oracle  : " + oracleAccuracy + " " + oracleNumCorrect + " / " + numExamples  
  }
}

object TestSemanticParserCli {
  def main(args: Array[String]): Unit = {
    (new TestSemanticParserCli()).run(args)
  }
}