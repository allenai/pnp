package org.allenai.dqa.matching

import scala.collection.JavaConverters._

import org.allenai.dqa.labeling.Diagram
import org.allenai.dqa.labeling.DiagramFeatures
import org.allenai.dqa.labeling.Part
import org.allenai.dqa.labeling.Point
import org.allenai.pnp.Env
import org.allenai.pnp.PnpModel

import com.jayantkrish.jklol.cli.AbstractCli
import com.jayantkrish.jklol.util.IoUtils

import edu.cmu.dynet._
import joptsimple.OptionParser
import joptsimple.OptionSet
import joptsimple.OptionSpec
import spray.json.DefaultJsonProtocol
import spray.json.pimpAny

class TestMatchingCli extends AbstractCli {
  
  var diagramsOpt: OptionSpec[String] = null
  var diagramFeaturesOpt: OptionSpec[String] = null
  var modelOpt: OptionSpec[String] = null
  var lossJson: OptionSpec[String] = null

  override def initializeOptions(parser: OptionParser): Unit = {
    diagramsOpt = parser.accepts("diagrams").withRequiredArg().ofType(classOf[String]).required()
    diagramFeaturesOpt = parser.accepts("diagramFeatures").withRequiredArg().ofType(classOf[String]).required()
    modelOpt = parser.accepts("model").withRequiredArg().ofType(classOf[String]).required()
    lossJson = parser.accepts("lossJson").withRequiredArg().ofType(classOf[String])
  }
  
  override def run(options: OptionSet): Unit = {
    Initialize.initialize()
    
    // Read and preprocess data
    val diagramFeatures = DiagramFeatures.fromJsonFile(options.valueOf(diagramFeaturesOpt)).map(
        x => (x.imageId, x)).toMap
    val diagramsAndLabels = Diagram.fromJsonFile(options.valueOf(diagramsOpt), diagramFeatures)
    
    // Sample diagram pairs of the same type to create
    // matching examples.
    val matchingExamples = TrainMatchingCli.sampleMatchingExamples(diagramsAndLabels, 30)
    
    // Read model
    val loader = new ModelLoader(options.valueOf(modelOpt))
    val model = PnpModel.load(loader)
    val matchingModel = MatchingModel.load(loader, model)
    loader.done()

    val losses = test(matchingExamples, matchingModel)

    if (options.has(lossJson)) {
      val jsons = losses.map(x => x.toJson(MyJsonProtocol.matchingLossFormat).compactPrint)
      IoUtils.writeLines(options.valueOf(lossJson), jsons.asJava)
    }
  }

  def test(examples: Seq[MatchingExample], matchingModel: MatchingModel): Seq[MatchingLoss] = {
    val beamSize = 5
    var numElementsCorrect = 0
    var numElements = 0
    var numDiagramsCorrect = 0
    
    val losses = for {
      x <- examples
    } yield {
      ComputationGraph.renew()
      val pnp = matchingModel.apply(x.source, x.target)
      
      val compGraph = matchingModel.model.getComputationGraph()
      val dist = pnp.beamSearch(beamSize, -1, Env.init, null, compGraph)

      val predicted = dist.executions(0).value
      val preprocessing = matchingModel.preprocess(x.source, x.target, compGraph)
      println(x.source.id + " -> " + x.target.id)
      println(x.source.id)
      for ((p, e) <- x.source.parts.zip(preprocessing.sourceFeatures)) {
        val v = ComputationGraph.incrementalForward(e).toSeq.mkString(" ")
        println("  " + p + " " + v)
      }
      println(x.target.id)
      for ((p, e) <- x.target.parts.zip(preprocessing.targetFeatures)) {
        val v = ComputationGraph.incrementalForward(e).toSeq.mkString(" ")
        println("  " + p + " " + v)
      }
      for (i <- 0 until preprocessing.matchScores.length) {
        println(preprocessing.matchScores(i)
            .map(x => ComputationGraph.incrementalForward(x).toFloat)
            .mkString(" "))
      }

      println("expected: " + x.label)
      
      for (ex <- dist.executions) {
        println("   " + ex.logProb.formatted("%02.3f") + " " + ex.value) 
      }

      if (predicted.equals(x.label)) {
        numDiagramsCorrect += 1
      }
      
      val intersection = predicted.targetToSourcePartMap.toSet.intersect(
          x.label.targetToSourcePartMap.toSet)
      numElementsCorrect += intersection.size
      numElements += predicted.targetToSourcePartMap.size

      val sourceDims = Point(x.source.width, x.source.height)
      val targetDims = Point(x.target.width, x.target.height)
      MatchingLoss(x.source.imageId, x.source.parts, sourceDims,
          x.target.imageId, x.target.parts, targetDims, 
          predicted.targetToSourcePartMap.toList)
    }

    val diagramAccuracy = numDiagramsCorrect.toDouble / examples.size
    println("Diagram accuracy: " + diagramAccuracy + " ( " + numDiagramsCorrect + " / " +  examples.size + " )")
    val partAccuracy = numElementsCorrect.toDouble / numElements
    println("Part accuracy: " + partAccuracy + " ( " + numElementsCorrect + " / " +  numElements + " )")
    
    losses
  } 
}

case class MatchingLoss(sourceImgId: String, sourceParts: Vector[Part], sourceDims: Point,
    targetImgId: String, targetParts: Vector[Part], targetDims: Point,
    matching: List[(Int, Int)]) {
}

object MyJsonProtocol extends DefaultJsonProtocol {
  implicit val pointFormat = jsonFormat2(Point)
  implicit val partFormat = jsonFormat3(Part)
  implicit val matchingLossFormat = jsonFormat7(MatchingLoss)
}

object TestMatchingCli {
  def main(args: Array[String]): Unit = {
    (new TestMatchingCli()).run(args)
  }
}