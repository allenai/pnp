package org.allenai.pnp.examples

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

import com.jayantkrish.jklol.ccg.CcgExample
import com.jayantkrish.jklol.ccg.cli.TrainSemanticParser
import com.jayantkrish.jklol.ccg.lambda.Type
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.cli.AbstractCli
import com.jayantkrish.jklol.cli.AbstractCli.CommonOptions
import com.jayantkrish.jklol.experiments.geoquery.GeoqueryUtil

import joptsimple.OptionParser
import joptsimple.OptionSet
import joptsimple.OptionSpec

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
    val lexicon = SemanticParser.generateLexicon(trainingData.map(_.getLogicalForm), typeDeclaration)
    
    println(lexicon.rootTypes)
    println(lexicon.typeTemplateMap)
    
    val parser = new SemanticParser(lexicon)
    
    validateLexicon(trainingData.map(_.getLogicalForm), parser, typeDeclaration)

    // TODO: 
    // 2. Parameterization
    // 
  }
  
  def validateLexicon(exprs: Seq[Expression2], parser: SemanticParser,
      typeDeclaration: TypeDeclaration): Unit = {
    for (e <- exprs) {
      // val dist = parser.generateLabeledExpression(ExpressionLabel.fromExpression(e, typeDeclaration))
      val dist = parser.generateExpression()
      val results = dist.beamSearch(100)
      
      if (results.size != 1) {
        println("ERROR: " + e + " " + results)
      }
    }
  }
}

object SemanticParserCli {
  def main(args: Array[String]): Unit = {
    (new SemanticParserCli()).run(args)
  }
}