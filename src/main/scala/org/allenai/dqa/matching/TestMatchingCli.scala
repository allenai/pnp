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
import edu.cmu.dynet.DyNetScalaHelpers.floatVectorToSeq
import edu.cmu.dynet.dynet_swig._
import joptsimple.OptionParser
import joptsimple.OptionSet
import joptsimple.OptionSpec
import spray.json.DefaultJsonProtocol
import spray.json.pimpAny
import org.allenai.dqa.labeling.DiagramLabel

class TestMatchingCli extends AbstractCli {
  
  var diagramsOpt: OptionSpec[String] = null
  var diagramFeaturesOpt: OptionSpec[String] = null
  var examplesOpt: OptionSpec[String] = null
  var modelOpt: OptionSpec[String] = null
  var beamSizeOpt: OptionSpec[Integer] = null
  var lossJson: OptionSpec[String] = null
  
  var enforceMatching: OptionSpec[Void] = null
  var globalNormalize: OptionSpec[Void] = null

  override def initializeOptions(parser: OptionParser): Unit = {
    diagramsOpt = parser.accepts("diagrams").withRequiredArg().ofType(classOf[String]).required()
    diagramFeaturesOpt = parser.accepts("diagramFeatures").withRequiredArg().ofType(classOf[String]).required()
    examplesOpt = parser.accepts("examples").withRequiredArg().ofType(classOf[String]).required()
    modelOpt = parser.accepts("model").withRequiredArg().ofType(classOf[String]).required()
    beamSizeOpt = parser.accepts("beamSize").withRequiredArg().ofType(classOf[Integer]).defaultsTo(5)
    lossJson = parser.accepts("lossJson").withRequiredArg().ofType(classOf[String])
    
    enforceMatching = parser.accepts("enforceMatching")
    globalNormalize = parser.accepts("globalNormalize")
  }
  
  override def run(options: OptionSet): Unit = {
    initialize(new DynetParams())
    
    // Read and preprocess data
    val diagramFeatures = DiagramFeatures.fromJsonFile(options.valueOf(diagramFeaturesOpt)).map(
        x => (x.imageId, x)).toMap
    val diagramsAndLabels = Diagram.fromJsonFile(options.valueOf(diagramsOpt), diagramFeatures)
    val diagramsMap = diagramsAndLabels.map(x => (x._1.id, x)).toMap

    // Read examples for training.
    val matchingExamples = MatchingExample.fromJsonFile(options.valueOf(examplesOpt), diagramsMap)
    // Sample diagram pairs of the same type to create
    // matching examples.
    // val matchingExamples = TrainMatchingCli.sampleMatchingExamples(diagramsAndLabels, 30)
    println(matchingExamples.length + " test examples.")

    // Read model
    val loader = new ModelLoader(options.valueOf(modelOpt))
    val model = PnpModel.load(loader)
    val matchingModel = MatchingModel.load(loader, model)
    loader.done()

    if (options.has(enforceMatching)) {
      matchingModel.config.matchIndependent = false
    }
    
    if (options.has(globalNormalize)) {
      matchingModel.model.locallyNormalized = false
    }

    val losses = test(matchingExamples, matchingModel, options.valueOf(beamSizeOpt))

    if (options.has(lossJson)) {
      val jsons = losses.map(x => x.toJson(MyJsonProtocol.matchingLossFormat).compactPrint)
      IoUtils.writeLines(options.valueOf(lossJson), jsons.asJava)
    }
  }

  def test(examples: Seq[MatchingExample], matchingModel: MatchingModel,
      beamSize: Int): Seq[MatchingLoss] = {
    var numElementsCorrect = 0
    var numElements = 0
    var numDiagramsCorrect = 0
    var numInvalidMatchings = 0
    
    val losses = for {
      x <- examples
    } yield {
      val pnp = matchingModel.apply(x.source, x.sourceLabel, x.target)
      
      val cg = ComputationGraph.getNew
      val compGraph = matchingModel.model.getComputationGraph(cg)
      val dist = pnp.beamSearch(beamSize, -1, Env.init, null, compGraph)

      val predicted = dist.executions(0).value
      val preprocessing = matchingModel.preprocess(x.source, x.sourceLabel, x.target,
          x.target.parts, compGraph)
      println(x.source.id + " -> " + x.target.id)
      println(x.source.id)
      for ((p, e) <- x.source.parts.zip(preprocessing.sourceFeatures)) {
        val v = as_vector(cg.incremental_forward(e.xy)).mkString(" ")
        println("  " + p + " " + v)
      }
      println(x.target.id)
      for ((p, e) <- x.target.parts.zip(preprocessing.targetFeatures)) {
        val v = as_vector(cg.incremental_forward(e.xy)).mkString(" ")
        println("  " + p + " " + v)
      }
      for (i <- 0 until preprocessing.matchScores.length) {
        println(preprocessing.matchScores(i).map(x => as_scalar(cg.incremental_forward(x))).mkString(" "))
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
      
      val valueSet = predicted.targetToSourcePartMap.values.toSet
      if (valueSet.size < predicted.targetToSourcePartMap.size) {
        numInvalidMatchings += 1
      }
      
      // TODO: Compute confusion matrix

      val sourceDims = Point(x.source.width, x.source.height)
      val targetDims = Point(x.target.width, x.target.height)
      MatchingLoss(x.source.imageId, x.source.parts, x.sourceLabel, sourceDims,
          x.target.imageId, x.target.parts, x.targetLabel, targetDims,
          predicted.targetToSourcePartMap.toList)
    }

    val invalidMatchings = numInvalidMatchings.toDouble / examples.size
    println("Invalid matchings: " + invalidMatchings + " ( " + numInvalidMatchings + " / " +  examples.size + " )")
    val diagramAccuracy = numDiagramsCorrect.toDouble / examples.size
    println("Diagram accuracy: " + diagramAccuracy + " ( " + numDiagramsCorrect + " / " +  examples.size + " )")
    val partAccuracy = numElementsCorrect.toDouble / numElements
    println("Part accuracy: " + partAccuracy + " ( " + numElementsCorrect + " / " +  numElements + " )")
    
    losses
  } 
}

case class MatchingLoss(sourceImgId: String, sourceParts: Vector[Part], sourceLabel: DiagramLabel,
    sourceDims: Point, targetImgId: String, targetParts: Vector[Part], targetLabel: DiagramLabel,
    targetDims: Point, matching: List[(Int, Int)]) {
}

object MyJsonProtocol extends DefaultJsonProtocol {
  implicit val pointFormat = jsonFormat2(Point)
  implicit val partFormat = jsonFormat3(Part)
  implicit val diagramLabelFormat = jsonFormat2(DiagramLabel)
  implicit val matchingLossFormat = jsonFormat9(MatchingLoss)
}

object TestMatchingCli {
  def main(args: Array[String]): Unit = {
    (new TestMatchingCli()).run(args)
  }
}