package org.allenai.pnp.semparse

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

import com.jayantkrish.jklol.ccg.CcgExample
import com.jayantkrish.jklol.ccg.cli.TrainSemanticParser
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.cli.AbstractCli
import com.jayantkrish.jklol.cli.AbstractCli.CommonOptions
import com.jayantkrish.jklol.experiments.geoquery.GeoqueryUtil

import joptsimple.OptionParser
import joptsimple.OptionSet
import joptsimple.OptionSpec
import com.jayantkrish.jklol.training.NullLogFunction
import org.allenai.pnp.Env
import org.allenai.pnp.CompGraph
import com.jayantkrish.jklol.util.IndexedList
import edu.cmu.dynet.ComputationGraph
import edu.cmu.dynet.dynet_swig._
import org.allenai.pnp.PpExample
import org.allenai.pnp.LoglikelihoodTrainer
import edu.cmu.dynet.SimpleSGDTrainer
import org.allenai.pnp.PpModel
import com.jayantkrish.jklol.ccg.lambda2.ExpressionSimplifier
import com.jayantkrish.jklol.ccg.lambda2.ExpressionComparator
import com.jayantkrish.jklol.ccg.lambda2.SimplificationComparator

class SemanticParserCli extends AbstractCli() {
  
  var trainingDataOpt: OptionSpec[String] = null
  
  override def initializeOptions(parser: OptionParser): Unit = {
    trainingDataOpt = parser.accepts("trainingData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',').required()
  }
  
  override def run(options: OptionSet): Unit = {
    myInitialize()
    
    val trainingData = ListBuffer[CcgExample]()
    for (filename <- options.valuesOf(trainingDataOpt).asScala) {
      trainingData ++= TrainSemanticParser.readCcgExamples(filename).asScala
    }

    val typeDeclaration = GeoqueryUtil.getTypeDeclaration()
    val simplifier = GeoqueryUtil.getExpressionSimplifier
    val comparator = new SimplificationComparator(simplifier)
    
    val preprocessed = trainingData.map(x =>
      new CcgExample(x.getSentence, x.getDependencies, x.getSyntacticParse, 
          simplifier.apply(x.getLogicalForm))) 
    
    val actionSpace = SemanticParser.generateActionSpace(preprocessed.map(_.getLogicalForm), typeDeclaration)
    val vocab = IndexedList.create[String]
    for (x <- preprocessed) {
      vocab.addAll(x.getSentence.getWords)
    }
    
    println(actionSpace.rootTypes)
    println(actionSpace.typeTemplateMap)
    
    val parser = new SemanticParser(actionSpace, vocab)
    
    // validateActionSpace(preprocessed, parser, typeDeclaration)
    val trainedModel = train(preprocessed, parser, typeDeclaration)
    // val trainedModel = parser.getModel
    test(preprocessed, parser, trainedModel, typeDeclaration, simplifier, comparator)
    // TODO: 
    // 2. Parameterization
    // 
  }
  
  def validateActionSpace(examples: Seq[CcgExample], parser: SemanticParser,
      typeDeclaration: TypeDeclaration): Unit = {
    val model = parser.getModel

    println("")
    for (e <- examples) {
      println(e.getSentence.getWords)
      println(e.getLogicalForm)

      val oracle = parser.generateExecutionOracle(e.getLogicalForm, typeDeclaration)
      val dist = parser.generateExpression(e.getSentence.getWords.asScala.toList)

      val cg = new ComputationGraph
      val results = dist.beamSearch(1, 50, Env.init, oracle,
          model.getInitialComputationGraph(cg), new NullLogFunction())
      if (results.executions.size != 1) {
        println("ERROR: " + e + " " + results)
      } else {
        println("OK   : " + e + " " + results.executions(0))
      }
      cg.delete

      /*
      // 
      val results = dist.beamSearch(100)
      
      if (results.size != 1) {
        println("ERROR: " + e + " " + results)
      }
      */
    }
  }
  
  def train(examples: Seq[CcgExample], parser: SemanticParser,
      typeDeclaration: TypeDeclaration): PpModel = {
    val ppExamples = examples map { x => 
      val words = x.getSentence.getWords
      val unconditional = parser.generateExpression(words.asScala.toList)
      val oracle = parser.generateExecutionOracle(x.getLogicalForm, typeDeclaration)
      PpExample(unconditional, unconditional, Env.init, oracle)
    }
    
    // Train model
    val model = parser.getModel
    val sgd = new SimpleSGDTrainer(model.model, 0.1f, 0.01f)
    val trainer = new LoglikelihoodTrainer(100, 100, model, sgd, new NullLogFunction())
    trainer.train(ppExamples.toList)

    model
  }
  
  def test(examples: Seq[CcgExample], parser: SemanticParser,
      model: PpModel, typeDeclaration: TypeDeclaration, simplifier: ExpressionSimplifier,
      comparator: ExpressionComparator): Unit = {
    println("")
    var numCorrect = 0
    for (e <- examples) {
      println(e.getSentence.getWords)
      println(e.getLogicalForm)

      val dist = parser.generateExpression(e.getSentence.getWords.asScala.toList)

      // TODO: null is wrong below
      val cg = new ComputationGraph
      val results = dist.beamSearch(100, 50, Env.init, null,
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

      /*
      // 
      val results = dist.beamSearch(100)
      
      if (results.size != 1) {
        println("ERROR: " + e + " " + results)
      }
      */
    }
    
    val accuracy = numCorrect.asInstanceOf[Double] / examples.length 
    println(accuracy + " " + numCorrect + " / " + examples.length)
  }
}

object SemanticParserCli {
  def main(args: Array[String]): Unit = {
    (new SemanticParserCli()).run(args)
  }
}