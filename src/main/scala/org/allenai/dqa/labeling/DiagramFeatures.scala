package org.allenai.dqa.labeling

import scala.io.Source

import edu.cmu.dynet._
import spray.json._
import spray.json.DefaultJsonProtocol._

/**
 * Features of points in a diagram. 
 */
case class DiagramFeatures(imageId: String, pointFeatures: Map[Point, FloatVector]) {

  def getFeatures(part: Part): FloatVector = {
    pointFeatures(part.coords)
  }

  def getFeatureMatrix(parts: Seq[Part]): Array[Expression] = {
    val expressions = for {
      part <- parts
    } yield {
      val features = getFeatures(part)
      Expression.input(Dim(features.size), features)
    }

    expressions.toArray
  }
}

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
      vec = p.fields("vec").asInstanceOf[JsArray].elements.map(x => x.convertTo[Float])
    } yield {
      (xy, new FloatVector(vec))
    }
    
    DiagramFeatures(imageId, pointFeatures.toMap)
  }
}