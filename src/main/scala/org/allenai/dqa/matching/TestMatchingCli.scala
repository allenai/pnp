package org.allenai.dqa.matching

import org.allenai.dqa.labeling.Diagram
import org.allenai.dqa.labeling.DiagramFeatures

import com.jayantkrish.jklol.cli.AbstractCli

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._
import joptsimple.OptionParser
import joptsimple.OptionSet
import joptsimple.OptionSpec
import org.allenai.pnp.PnpModel
import org.allenai.pnp.Env


class TestMatchingCli extends AbstractCli {
  
  var diagramsOpt: OptionSpec[String] = null
  var diagramFeaturesOpt: OptionSpec[String] = null
  var modelOpt: OptionSpec[String] = null

  override def initializeOptions(parser: OptionParser): Unit = {
    diagramsOpt = parser.accepts("diagrams").withRequiredArg().ofType(classOf[String]).required()
    diagramFeaturesOpt = parser.accepts("diagramFeatures").withRequiredArg().ofType(classOf[String]).required()
    modelOpt = parser.accepts("model").withRequiredArg().ofType(classOf[String]).required()
  }
  
  override def run(options: OptionSet): Unit = {
    initialize(new DynetParams())
    
    // Read and preprocess data
    val diagramFeatures = DiagramFeatures.fromJsonFile(options.valueOf(diagramFeaturesOpt)).map(
        x => (x.imageId, x)).toMap
    val diagramsAndLabels = Diagram.fromJsonFile(options.valueOf(diagramsOpt), diagramFeatures)
    
    // Sample diagram pairs of the same type to create
    // matching examples.
    val matchingExamples = TrainMatchingCli.sampleMatchingExamples(diagramsAndLabels, 10)
    
    // Read model
    val loader = new ModelLoader(options.valueOf(modelOpt))
    val model = PnpModel.load(loader)
    val matchingModel = MatchingModel.load(loader, model)
    loader.done()
    
    test(matchingExamples, matchingModel)
  }

  def test(examples: Seq[MatchingExample], matchingModel: MatchingModel): Unit = {
    
    var numElementsCorrect = 0
    var numElements = 0
    var numDiagramsCorrect = 0
    for (x <- examples) {
      val pnp = matchingModel.apply(x.source, x.target)
      
      val cg = ComputationGraph.getNew
      val dist = pnp.beamSearch(5, -1, Env.init, null, 
          matchingModel.model.getComputationGraph(cg))
          
      val predicted = dist.executions(0).value
      println(x.source.id + " -> " + x.target.id)
      println("  " + x.label)
      println("  " + predicted)
      
      if (predicted.equals(x.label)) {
        numDiagramsCorrect += 1
      }
      
      val intersection = predicted.targetToSourcePartMap.toSet.intersect(
          x.label.targetToSourcePartMap.toSet)
      numElementsCorrect += intersection.size
      numElements += predicted.targetToSourcePartMap.size
    }
    
    val diagramAccuracy = numDiagramsCorrect.toDouble / examples.size
    println("Diagram accuracy: " + diagramAccuracy + " ( " + numDiagramsCorrect + " / " +  examples.size + " )")
    val partAccuracy = numElementsCorrect.toDouble / numElements
    println("Part accuracy: " + partAccuracy + " ( " + numElementsCorrect + " / " +  numElements + " )")
  } 
}

object TestMatchingCli {
  def main(args: Array[String]): Unit = {
    (new TestMatchingCli()).run(args)
  }
}