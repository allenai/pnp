package org.allenai.dqa.labeling

import scala.io.Source

import edu.cmu.dynet._
import edu.cmu.dynet.DyNetScalaHelpers._
import edu.cmu.dynet.dynet_swig._
import spray.json._
import spray.json.DefaultJsonProtocol._

/**
 * Features of points in a diagram. 
 */
case class DiagramFeatures(imageId: String, pointFeatures: Map[Point, PointFeatures]) {

  def getFeatures(part: Part): PointFeatures = {
    pointFeatures(part.coords)
  }

  def getFeatureMatrix(parts: Seq[Part], cg: ComputationGraph): Array[PointExpressions] = {
    val expressions = for {
      part <- parts
    } yield {
      val features = pointFeatures(part.coords)
      val xy = input(cg, Seq(features.xy.size().asInstanceOf[Int]), features.xy)
      val matching = input(cg, Seq(features.matching.size().asInstanceOf[Int]), features.matching)
      val vgg0 = input(cg, Seq(features.vgg0.size().asInstanceOf[Int]), features.vgg0)
      val vgg1 = input(cg, Seq(features.vgg1.size().asInstanceOf[Int]), features.vgg1)
      val vgg2 = input(cg, Seq(features.vgg2.size().asInstanceOf[Int]), features.vgg2)
      val vggAll = input(cg, Seq(features.vggAll.size().asInstanceOf[Int]), features.vggAll)
      PointExpressions(xy, matching, vgg0, vgg1, vgg2, vggAll)
    }
    expressions.toArray
  }
}

case class PointFeatures(xy: FloatVector, matching: FloatVector,
    vgg0: FloatVector, vgg1: FloatVector, vgg2: FloatVector,
    vggAll: FloatVector)
case class PointExpressions(xy: Expression, matching: Expression,
    vgg0: Expression, vgg1: Expression, vgg2: Expression,
    vggAll: Expression)

object DiagramFeatures {
  
  def fromJsonFile(filename: String): Array[DiagramFeatures] = {
    val lines = Source.fromFile(filename).getLines
    lines.map(fromJsonLine(_)).toArray
  }
  
  def fromJsonLine(line: String): DiagramFeatures = {
    val js = line.parseJson.asJsObject
    val imageId = js.fields("imageId").convertTo[String]
    
    val pointJsons = js.fields("points").asInstanceOf[JsArray]
    
    val pointFeatures = for {
      pointJson <- pointJsons.elements
      p = pointJson.asJsObject
      xy = p.fields("xy") match {
        case JsArray(Vector(JsNumber(x), JsNumber(y))) => Point(x.toInt, y.toInt)
        case _ => deserializationError("Array of x/y coordinates expected")
      }
      
      xyVec = new FloatVector(p.fields("xy_vec").asInstanceOf[JsArray].elements.map(x => x.convertTo[Float]))
      matchingVec = new FloatVector(p.fields("matching_vec").asInstanceOf[JsArray].elements.map(x => x.convertTo[Float]))
      vgg0Vec = new FloatVector(p.fields("vgg_0_vec").asInstanceOf[JsArray].elements.map(x => x.convertTo[Float]))
      vgg1Vec = new FloatVector(p.fields("vgg_1_vec").asInstanceOf[JsArray].elements.map(x => x.convertTo[Float]))
      vgg2Vec = new FloatVector(p.fields("vgg_2_vec").asInstanceOf[JsArray].elements.map(x => x.convertTo[Float]))
      vggAll = new FloatVector(vgg0Vec.toSeq ++ vgg1Vec.toSeq ++ vgg2Vec.toSeq)
    } yield {
      (xy, PointFeatures(xyVec, matchingVec, vgg0Vec, vgg1Vec, vgg2Vec, vggAll))
    }

    DiagramFeatures(imageId, pointFeatures.toMap)
  }
}