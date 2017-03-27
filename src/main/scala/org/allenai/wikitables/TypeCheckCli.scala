package org.allenai.wikitables

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

import org.allenai.pnp.semparse.SemanticParserUtils

import com.jayantkrish.jklol.ccg.CcgExample
import com.jayantkrish.jklol.ccg.cli.TrainSemanticParser
import com.jayantkrish.jklol.cli.AbstractCli
import com.jayantkrish.jklol.experiments.geoquery.GeoqueryUtil

import joptsimple.OptionParser
import joptsimple.OptionSet
import joptsimple.OptionSpec

class TypeCheckCli extends AbstractCli() {
  
  var dataOpt: OptionSpec[String] = null
  
  override def initializeOptions(parser: OptionParser): Unit = {
    dataOpt = parser.accepts("data").withRequiredArg().ofType(classOf[String])
        .withValuesSeparatedBy(',').required()
  }
  
  override def run(options: OptionSet): Unit = {
    //val typeDeclaration = GeoqueryUtil.getSimpleTypeDeclaration()
    val typeDeclaration = new WikiTablesTypeDeclaration()
    
    // Read and data
    val trainingData = ListBuffer[CcgExample]()
    for (filename <- options.valuesOf(dataOpt).asScala) {
      trainingData ++= TrainSemanticParser.readCcgExamples(filename).asScala
    }

    println(trainingData.size + " training examples")

    trainingData.map(x => SemanticParserUtils.validateTypes(x.getLogicalForm, typeDeclaration))
  }
}

object TypeCheckCli {
   def main(args: Array[String]): Unit = {
    (new TypeCheckCli()).run(args)
  }
}
