package org.allenai.dqa.labeling

import scala.io.Source

import spray.json.DefaultJsonProtocol._
import spray.json.JsArray
import spray.json.JsNumber
import spray.json.JsObject
import spray.json.deserializationError
import spray.json.pimpString
import scala.util.Random

/**
 * A diagram marked with a collection of parts. Each
 * part has an x/y coordinate and a text label (e.g. "A") 
 */
case class Diagram(id: String, imageId: String, width: Int, height: Int,
    parts: Vector[Part], features: DiagramFeatures)

/**
 * A part of a diagram.
 */
case class Part(id: String, ind: Int, coords: Point)

/**
 * An x/y point in a diagram. 
 */
case class Point(x: Int, y: Int)

/**
 * A label for a diagram. The label includes a type for
 * the entire diagram (e.g., "car") along with labels for
 * each part (e.g., "wheel"). The indexes of {@code partLabels}  
 * correspond to {@code part.ind}.
 */
case class DiagramLabel(diagramType: String, partLabels: Vector[String])

object Diagram {
  
  def fromJsonFile(filename: String, features: Map[String, DiagramFeatures]
    ): Array[(Diagram, DiagramLabel)] = {
    val lines = Source.fromFile(filename).getLines
    lines.map(fromJsonLine(_, features)).toArray
  }

  def fromJsonLine(line: String, features: Map[String, DiagramFeatures]
    ): (Diagram, DiagramLabel) = {
    val js = line.parseJson.asJsObject
    val diagramLabel = js.fields("label").convertTo[String]
    val diagramId = js.fields("id").convertTo[String]
    val imageId = js.fields("imageId").convertTo[String]
    val width = js.fields("width").convertTo[Int]
    val height = js.fields("height").convertTo[Int]
    
    // val pointJsons = Random.shuffle(js.fields("points").asInstanceOf[JsArray].elements)
    val pointJsons = js.fields("points").asInstanceOf[JsArray].elements

    val labeledParts = for {
      (pointJson, i) <- pointJsons.zipWithIndex
      p = pointJson.asJsObject
      id = p.fields("textId").convertTo[String]
      label = p.fields("label").convertTo[String]
      xy = p.fields("xy") match {
        case JsArray(Vector(JsNumber(x), JsNumber(y))) => Point(x.toInt, y.toInt)
        case _ => deserializationError("Array of x/y coordinates expected")
      }
    } yield {
      (Part(id, i, xy),  label)
    }

    val f = features(imageId)

    (Diagram(diagramId, imageId, width, height, labeledParts.map(_._1), f),
        (DiagramLabel(diagramLabel, labeledParts.map(_._2))))
  }
}