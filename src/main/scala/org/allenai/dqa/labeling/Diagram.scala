package org.allenai.dqa.labeling

import scala.collection.JavaConverters._

import spray.json._
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import com.jayantkrish.jklol.util.IoUtils

case class Diagram(id: String, parts: Vector[Part]) {
  
}

case class DiagramLabel(diagramType: String, partLabels: Vector[String]) { 
}

case class Part(id: String, coords: (Int, Int)) {
  
}

object Diagram {
  
  def fromJsonFile(filename: String): Array[(Diagram, DiagramLabel)] = {
    val lines = IoUtils.readLines(filename).asScala
    
    lines.map(x => fromJsonObject(x.parseJson.asJsObject)).toArray
  }

  def fromJsonObject(js: JsObject): (Diagram, DiagramLabel) = {
    val diagramLabel = js.fields("label").toString
    val diagramId = js.fields("id").toString
    val imageId = js.fields("imageId").toString
    
    val pointJsons = js.fields("points").asInstanceOf[JsArray]
    
    val labeledParts = for {
      pointJson <- pointJsons.elements
      p = pointJson.asJsObject
      id = p.fields("textId").toString
      label = p.fields("label").toString
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