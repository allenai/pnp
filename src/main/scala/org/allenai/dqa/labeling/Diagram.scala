package org.allenai.dqa.labeling

import scala.io.Source

import spray.json.DefaultJsonProtocol._
import spray.json.JsArray
import spray.json.JsNumber
import spray.json.JsObject
import spray.json.deserializationError
import spray.json.pimpString

/**
 * A diagram marked with a collection of parts. Each
 * part has an x/y coordinate and a text label (e.g. "A") 
 */
case class Diagram(id: String, parts: Vector[Part])

/**
 * A part of a diagram.
 */
case class Part(id: String, ind: Int, coords: (Int, Int))

/**
 * A label for a diagram. The label includes a type for
 * the entire diagram (e.g., "car") along with labels for
 * each part (e.g., "wheel"). The indexes of {@code partLabels}  
 * correspond to indexes of {@code parts}.
 */
case class DiagramLabel(diagramType: String, partLabels: Vector[String])

object Diagram {
  
  def fromJsonFile(filename: String): Array[(Diagram, DiagramLabel)] = {
    val lines = Source.fromFile(filename).getLines
    lines.map(fromJsonLine(_)).toArray
  }

  def fromJsonLine(line: String): (Diagram, DiagramLabel) = {
    val js = line.parseJson.asJsObject
    val diagramLabel = js.fields("label").convertTo[String]
    val diagramId = js.fields("id").convertTo[String]
    val imageId = js.fields("imageId").convertTo[String] 
    
    val pointJsons = js.fields("points").asInstanceOf[JsArray]
    
    val labeledParts = for {
      (pointJson, i) <- pointJsons.elements.zipWithIndex
      p = pointJson.asJsObject
      id = p.fields("textId").convertTo[String]
      label = p.fields("label").convertTo[String]
      xy = p.fields("xy") match {
        case JsArray(Vector(JsNumber(x), JsNumber(y))) => (x.toInt, y.toInt)
        case _ => deserializationError("Array of x/y coordinates expected")
      }
    } yield {
      (Part(id, i, xy),  label)
    }
    
    (Diagram(diagramId, labeledParts.map(_._1)),
        (DiagramLabel(diagramLabel, labeledParts.map(_._2)))) 
  }
}