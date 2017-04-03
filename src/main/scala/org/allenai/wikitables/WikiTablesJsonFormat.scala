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
}