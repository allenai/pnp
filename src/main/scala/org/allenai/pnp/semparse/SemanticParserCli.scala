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

class SemanticParserCli extends AbstractCli(
  CommonOptions.MAP_REDUCE, CommonOptions.STOCHASTIC_GRADIENT, CommonOptions.LBFGS
) {
  
  var trainingDataOpt: OptionSpec[String] = null
  
  override def initializeOptions(parser: OptionParser): Unit = {
    trainingDataOpt = parser.accepts("trainingData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',').required()
  }
  
  override def run(options: OptionSet): Unit = {
    val trainingData = ListBuffer[CcgExample]()
    for (filename <- options.valuesOf(trainingDataOpt).asScala) {
      trainingData ++= TrainSemanticParser.readCcgExamples(filename).asScala
    }

    val typeDeclaration = GeoqueryUtil.getTypeDeclaration()
    val simplifier = GeoqueryUtil.getExpressionSimplifier
    
    val lfs = trainingData.map(x => simplifier.apply(x.getLogicalForm))
    val actionSpace = SemanticParser.generateActionSpace(lfs, typeDeclaration)
    
    println(actionSpace.rootTypes)
    println(actionSpace.typeTemplateMap)
    
    val parser = new SemanticParser(actionSpace)
    
    validateActionSpace(lfs, parser, typeDeclaration)

    // TODO: 
    // 2. Parameterization
    // 
  }
  
  def validateActionSpace(exprs: Seq[Expression2], parser: SemanticParser,
      typeDeclaration: TypeDeclaration): Unit = {
    println("")
    for (e <- exprs) {
      println(e)

      val oracle = parser.generateExecutionOracle(e, typeDeclaration)
      val dist = parser.generateExpression()
      val results = dist.beamSearch(1, Env.init, oracle, CompGraph.empty, new NullLogFunction())
      if (results.executions.size != 1) {
        println("ERROR: " + e + " " + results)
      } else {
        println("OK   : " + e + " " + results.executions(0))
      }

      /*
      // 
      val results = dist.beamSearch(100)
      
      if (results.size != 1) {
        println("ERROR: " + e + " " + results)
      }
      */
    }
  }
}

object SemanticParserCli {
  def main(args: Array[String]): Unit = {
    (new SemanticParserCli()).run(args)
  }
}