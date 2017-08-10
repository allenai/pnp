package org.allenai.wikitables

import scala.collection.JavaConverters._
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.nlpannotation.AnnotatedSentence
import edu.stanford.nlp.sempre._
import edu.stanford.nlp.sempre.tables.TableKnowledgeGraph
import fig.basic.LispTree

import scala.util.Try

/**
 * Similar to CcgExample, except that this supports multiple logical forms per example, and also stores the
 * context and target values from Sempre to execute and evaluate formulas.
 * This currently does not hold a syntactic parse or dependencies. But we may need them in the future.
 */
case class WikiTablesExample(
  id: String,
  sentence: AnnotatedSentence,
  goldLogicalForm: Option[Expression2],
  possibleLogicalForms: Set[Expression2],
  tableString: String,
  targetValue: Option[Value]
) {

  val logicalForms = goldLogicalForm match {
    case None => possibleLogicalForms
    case Some(lf) => Set(lf)
  }

  /**
   * Loads the knowledge graph into memory, from the `tableString` constructor argument.  We do not
   * keep a reference to this object, so it can get garbage collected when the caller is done with
   * the object.  TODO(matt): we might want to reconsider this, as it's slower than I thought it
   * was.
   */
  def getContext(): ContextValue = {
    val lispTree = LispTree.proto.parseFromString(tableString)
    val contextTree = lispTree.child(1)
    if (contextTree.child(0).value.equals("table")) {
      // This means we are dealing with the actual table data itself, not a pointer to a file containing it.
      // We'll have to extract the relevant info from the lisp tree and make a contextvalue.
      val tableValue = new TableValue(contextTree)
      // First row in the table
      val headerString: String = tableValue.header.toArray.mkString(",")
      val dataString: String = tableValue.rows.asScala.map(row =>
        row.asScala.map(cell =>
          cell.asInstanceOf[NameValue].description).mkString(",")).mkString("###")
      val fullTableDataString = List(headerString, dataString).mkString("###")
      new ContextValue(new TableKnowledgeGraph(null, new TableStringReader(fullTableDataString)))
    } else {
      new ContextValue(LispTree.proto.parseFromString(tableString))
    }
  }
  
  def executeFormula(pnpFormula: Expression2): Option[Value] = {
    // Sempre represents lambda expressions differently. We changed them when reading the examples. Changing
    // them back for execution.
    val expressionStringOpt = WikiTablesUtil.toSempreLogicalForm(pnpFormula);
    if (!expressionStringOpt.isDefined) {
      System.err.println("Ill-formed formula: " + pnpFormula);
      None;
    } else {
      try {
        val sempreFormula = Formula.fromString(expressionStringOpt.get);
        Some(WikiTablesDataProcessor.executeFormula(sempreFormula, getContext(), null))
      } catch {
        case e: Exception => {
          System.err.println("Bad formula: " + expressionStringOpt.get);
          None
        }
      }
    }
  }
  
  def isValueCorrect(predicted: Value): Boolean = {
    try {
      WikiTablesDataProcessor.isValueCorrect(predicted, targetValue.orNull, null);
    } catch {
      case e: Exception => {
        System.err.println("Bad value: " + predicted);
        false;
      }
    }
  }

  def isFormulaCorrect(pnpFormula: Expression2): Boolean = {
    // Sempre represents lambda expressions differently. We changed them when reading the examples. Changing
    // them back for execution.
    val expressionStringOpt = WikiTablesUtil.toSempreLogicalForm(pnpFormula);
    if (!expressionStringOpt.isDefined) {
      System.err.println("Ill-formed formula: " + pnpFormula);
      false;
    } else {
      try {
        val sempreFormula = Formula.fromString(expressionStringOpt.get);
        WikiTablesDataProcessor.isFormulaCorrect(sempreFormula, getContext(), targetValue.orNull, null);
      } catch {
        case e: Exception => {
          System.err.println("Bad formula: " + expressionStringOpt.get);
          false;
        }
      }
    }
  }

  override def toString() = {
    sentence + " [\n" + logicalForms.map(_.toString).mkString(" ") + "\n]"
  }
}

case class RawExample(ex: WikiTablesExample, linking: RawEntityLinking, table: Table)