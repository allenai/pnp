package org.allenai.dqa.labeling

import scala.io.Source

import edu.cmu.dynet._
import spray.json._
import spray.json.DefaultJsonProtocol._

/**
 * Features of points in a diagram. 
 */
case class DiagramFeatures(imageId: String, pointFeatures: Map[Point, PointFeatures]) {

  def getFeatures(part: Part): PointFeatures = {
    pointFeatures(part.coords)
  }

  def getFeatureMatrix(parts: Seq[Part]): Array[PointExpressions] = {
    val expressions = for {
      part <- parts
    } yield {
      val features = pointFeatures(part.coords)
      val xy = Expression.input(Dim(features.xy.size), features.xy)
      val matching = Expression.input(Dim(features.matching.size), features.matching)
      val vgg0 = Expression.input(Dim(features.vgg0.size), features.vgg0)
      val vgg1 = Expression.input(Dim(features.vgg1.size), features.vgg1)
      val vgg2 = Expression.input(Dim(features.vgg2.size), features.vgg2)
      val vggAll = Expression.input(Dim(features.vggAll.size), features.vggAll)
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
      
      xyVec = new FloatVector(p.fields("xy_vec").asInstanceOf[JsArray].elements.map(
          x => x.convertTo[Float]))
      matchingVec = new FloatVector(p.fields("matching_vec").asInstanceOf[JsArray].elements.map(
          x => x.convertTo[Float]))
      vgg0Vec = new FloatVector(p.fields("vgg_0_vec").asInstanceOf[JsArray].elements.map(
          x => x.convertTo[Float]))
      vgg1Vec = new FloatVector(p.fields("vgg_1_vec").asInstanceOf[JsArray].elements.map(
          x => x.convertTo[Float]))
      vgg2Vec = new FloatVector(p.fields("vgg_2_vec").asInstanceOf[JsArray].elements.map(
          x => x.convertTo[Float]))
      vggAll = new FloatVector(vgg0Vec.toSeq ++ vgg1Vec.toSeq ++ vgg2Vec.toSeq)
    } yield {
      (xy, PointFeatures(xyVec, matchingVec, vgg0Vec, vgg1Vec, vgg2Vec, vggAll))
    }

    DiagramFeatures(imageId, pointFeatures.toMap)
  }
}