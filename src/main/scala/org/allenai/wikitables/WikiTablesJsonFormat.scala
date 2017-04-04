package org.allenai.wikitables

import edu.stanford.nlp.sempre.Formula
import spray.json._
import org.allenai.pnp.semparse.Span

/**
 * Protocol for serializing wikitables-related objects to JSON.
 */
object WikiTablesJsonFormat extends DefaultJsonProtocol {
  implicit val spanFormat = jsonFormat2(Span)

  implicit object FormulaJsonFormat extends RootJsonFormat[Formula] {
    def write(f: Formula) = JsString(f.toString())

    def read(value: JsValue) = value match {
      case JsString(s) => Formula.fromString(s)
      case _ => deserializationError("formula expected")
    }
  }

  implicit val entityLinkingFormat = jsonFormat2(RawEntityLinking.apply)
  
  implicit val nerTagFormat = jsonFormat2(NerTag)
  implicit val cellFormat = jsonFormat6(Cell.apply)
  implicit val columnFormat = jsonFormat6(Column.apply)
  
  implicit val tableJsonFormat = jsonFormat3(TableJson.apply)

  implicit object TableFormat extends RootJsonFormat[Table] {
    def write(t: Table) = t.toTableJson.toJson
    def read(value: JsValue) = Table.fromTableJson(value.convertTo[TableJson]) 
  }
}