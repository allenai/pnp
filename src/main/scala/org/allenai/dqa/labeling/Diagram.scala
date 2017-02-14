package org.allenai.dqa.labeling

import scala.io.Source

import spray.json.DefaultJsonProtocol._
import spray.json.JsArray
import spray.json.JsNumber
import spray.json.JsObject
import spray.json.deserializationError
import spray.json.pimpString

case class Diagram(id: String, parts: Vector[Part]) {
  
}

case class DiagramLabel(diagramType: String, partLabels: Vector[String]) { 
}

case class Part(id: String, coords: (Int, Int)) {
  
}

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
      pointJson <- pointJsons.elements
      p = pointJson.asJsObject
      id = p.fields("textId").convertTo[String]
      label = p.fields("label").convertTo[String]
      xy = p.fields("xy") match {
        case JsArray(Vector(JsNumber(x), JsNumber(y))) => (x.toInt, y.toInt)
        case _ => deserializationError("Array of x/y coordinates expected")
      }
    } yield {
      (Part(id, xy),  label)
    }
    
    (Diagram(diagramId, labeledParts.map(_._1)),
        (DiagramLabel(diagramLabel, labeledParts.map(_._2)))) 
  }
}