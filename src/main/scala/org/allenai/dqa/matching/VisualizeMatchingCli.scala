package org.allenai.dqa.matching

import scala.collection.JavaConverters._

import org.allenai.dqa.labeling.Diagram
import org.allenai.dqa.labeling.DiagramFeatures
import org.allenai.dqa.labeling.Part
import org.allenai.dqa.labeling.Point
import org.allenai.dqa.labeling.PointFeatures
import org.allenai.pnp.PnpModel

import com.jayantkrish.jklol.cli.AbstractCli

import edu.cmu.dynet._
import edu.cmu.dynet.DyNetScalaHelpers._
import edu.cmu.dynet.dynet_swig._
import joptsimple.OptionParser
import joptsimple.OptionSet
import joptsimple.OptionSpec
import scala.util.Random
import org.allenai.pnp.Pnp
import org.allenai.dqa.labeling.DiagramLabel

class VisualizeMatchingCli extends AbstractCli {
  var diagramsOpt: OptionSpec[String] = null
  var diagramFeaturesOpt: OptionSpec[String] = null
  var modelOpt: OptionSpec[String] = null
  
  var sourceOpt: OptionSpec[String] = null
  var targetOpt: OptionSpec[String] = null
  var labelsToMatch: OptionSpec[String] = null
  var sourcePartOpt: OptionSpec[String] = null
  var numGridOpt: OptionSpec[Integer] = null

  override def initializeOptions(parser: OptionParser): Unit = {
    diagramsOpt = parser.accepts("diagrams").withRequiredArg().ofType(classOf[String]).required()
    diagramFeaturesOpt = parser.accepts("diagramFeatures").withRequiredArg().ofType(classOf[String]).required()
    modelOpt = parser.accepts("model").withRequiredArg().ofType(classOf[String]).required()
    
    sourceOpt = parser.accepts("source").withRequiredArg().ofType(classOf[String]).required()
    targetOpt = parser.accepts("target").withRequiredArg().ofType(classOf[String]).required()
    labelsToMatch = parser.accepts("labelsToMatch").withRequiredArg().ofType(classOf[String])
      .withValuesSeparatedBy(",").required()
    sourcePartOpt = parser.accepts("sourcePart").withRequiredArg().ofType(classOf[String]).required()
    numGridOpt = parser.accepts("numGrid").withRequiredArg().ofType(classOf[Integer]).defaultsTo(10)
  }
  
  override def run(options: OptionSet): Unit = {
    initialize(new DynetParams())

    // Read and preprocess data
    val diagramFeatures = DiagramFeatures.fromJsonFile(options.valueOf(diagramFeaturesOpt)).map(
        x => (x.imageId, x)).toMap
    val diagramsAndLabels = Diagram.fromJsonFile(options.valueOf(diagramsOpt), diagramFeatures)
    val diagramsMap = diagramsAndLabels.map(x => (x._1.id, x)).toMap

    // Read model
    val loader = new ModelLoader(options.valueOf(modelOpt))
    val model = PnpModel.load(loader)
    val matchingModel = MatchingModel.load(loader, model)
    loader.done()

    val (source, sourceLabel) = diagramsMap(options.valueOf(sourceOpt))
    val (target, targetLabel) = diagramsMap(options.valueOf(targetOpt))

    val matching = for {
      label <- options.valuesOf(labelsToMatch).asScala 
    } yield {
      val targetInd = targetLabel.partLabels.indexOf(label)
      val sourceInd = sourceLabel.partLabels.indexOf(label)
      (target.parts(targetInd), source.parts(sourceInd)) 
    }

    val numGrid = options.valueOf(numGridOpt)
    val sourcePart = source.parts(sourceLabel.partLabels.indexOf(options.valueOf(sourcePartOpt)))
    val scores = getGlobalScores(matching, source, sourceLabel, target, sourcePart,
        matchingModel, numGrid)
    
    val sortedScores = scores.toList.sortBy(p => (p._1.y, p._1.x))
    val matrix = (for {
      i <- 0 until numGrid
    } yield {
      sortedScores.slice(i * numGrid, (i + 1) * numGrid).map(_._2).toArray
    }).toArray

    println("[")
    println(matrix.map(row => "[" + row.map(_.formatted("%02.3f")).mkString(", ") + "]").mkString(",\n"))
    println("]")
  }
  
  def augmentDiagramParts(diagram: Diagram, numGridPoints: Int): Diagram = {
    // Extend the target diagram with many Parts in a grid.
    val newParts = for {
      i <- 0 until numGridPoints
      j <- 0 until numGridPoints
    } yield {
      val x = ((j + 0.5) * (diagram.width.toFloat / numGridPoints)).toInt
      val y = ((i + 0.5) * (diagram.height.toFloat / numGridPoints)).toInt
      val point = Point(x, y)

      val partInd = diagram.parts.length + (i * numGridPoints) + j
      Part("n/a", partInd, point) 
    }
    
    val features = diagram.features.pointFeatures(diagram.parts(0).coords)
    val newFeatures = newParts.map{ part =>
      val point = part.coords
      val normX = point.x.toFloat / diagram.width
      val normY = point.y.toFloat / diagram.height
      
      val xyFeatures = new FloatVector(List(normX, normY))
      val matchingFeatures = new FloatVector(List.fill(features.matching.length)(0f))
      val vgg0 = new FloatVector(List.fill(features.vgg0.length)(0.0f))
      val vgg1 = new FloatVector(List.fill(features.vgg1.length)(0.0f))
      val vgg2 = new FloatVector(List.fill(features.vgg2.length)(0.0f))
      val vggAll = new FloatVector(List.fill(features.vggAll.length)(0.0f))
      val pointFeatures = PointFeatures(xyFeatures, matchingFeatures, vgg0, vgg1, vgg2, vggAll)

      (point, pointFeatures)
    }.toMap

    val newDiagramFeatures = DiagramFeatures(diagram.features.imageId,
        diagram.features.pointFeatures ++ newFeatures) 
    
    Diagram(diagram.id, diagram.imageId, diagram.width, diagram.height,
        diagram.parts ++ newParts, newDiagramFeatures)
  }

  def getGlobalScores(matching: Seq[(Part, Part)], source: Diagram, sourceLabel: DiagramLabel,
      target: Diagram, sourcePart: Part, model: MatchingModel, numGridPoints: Int): Map[Point, Float] = {
    val augmentedTarget = augmentDiagramParts(target, numGridPoints)
    val gridParts = augmentedTarget.parts.drop(target.parts.length)

    val computationGraph = ComputationGraph.getNew
    val cg = model.model.getComputationGraph(computationGraph)
    val preprocessing = model.preprocess(source, sourceLabel, augmentedTarget, cg)

    val matchingList = matching.toList
    val matchingScore = model.getNnGlobalScore(matchingList, cg, preprocessing)

    val partScoreMap = gridParts.map {
      part =>
      val candidateMatching = (part, sourcePart) :: matchingList
      val candidateScore = model.getNnGlobalScore(candidateMatching, cg, preprocessing)
      
      val scoreDelta = as_scalar(cg.cg.incremental_forward(candidateScore - matchingScore))
      (part.coords, scoreDelta)
    }

    partScoreMap.toMap
  }
}

object VisualizeMatchingCli {
  def main(args: Array[String]): Unit = {
    (new VisualizeMatchingCli()).run(args)
  }
}
