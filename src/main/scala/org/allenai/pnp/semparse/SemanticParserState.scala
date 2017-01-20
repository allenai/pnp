package org.allenai.pnp.semparse

import java.util.Arrays

import com.google.common.base.Preconditions
import com.jayantkrish.jklol.ccg.lambda.Type
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import edu.cmu.dynet.Expression

/** State of the semantic parser during expression generation.
  * Each hole generated during parsing is assigned a unique id.
  * When a template is applied to the hole, the corresponding 
  * partial expression is mapped to the hole id. Unfilled holes
  * are stored in a list that tracks which portions of the
  * expression have yet to be generated.
  */
case class SemanticParserState(val parts: Map[Int, ExpressionPart],
    val unfilledHoleIds: List[(Int, Type, Scope)], val nextId: Int,
    val numActions: Int, val templates: List[Template], val attentions: List[Expression]) {

  def decodeExpression(partId: Int): Expression2 = {
    val part = parts(partId)

    var expr = part.expr
    for (i <- 1 to part.holes.length) {
      val ind = part.holes.length - i
      var subexpr = decodeExpression(part.holeIds(ind))
      expr = expr.substitute(part.holes(ind), subexpr)
    }

    expr
  }
  
  def decodeExpression: Expression2 = {
    Preconditions.checkState(unfilledHoleIds.length == 0)
    decodeExpression(0)
  }
  
  def addAttention(e: Expression): SemanticParserState = {
    SemanticParserState(parts, unfilledHoleIds, nextId, numActions,
        templates, e :: attentions)
  }
  
  def getTemplates: Array[Template] = {
    templates.reverse.toArray
  }
  
  def getAttentions: Array[Expression] = {
    attentions.reverse.toArray
  }
}

object SemanticParserState {

  /** The start state of a semantic parser for generating
    * an expression of type t.
    */
  def start(t: Type): SemanticParserState = {
    val scope = Scope(List.empty)    
    SemanticParserState(Map.empty, List((0, t, scope)), 1, 0, List(), List()) 
  }
}

case class ExpressionPart(val expr: Expression2,
    val holes: Array[Int], val holeIds: Array[Int]) {
  Preconditions.checkArgument(holes.length == holeIds.length)
  
  override def toString: String = {
    "ExpressionPart(" + expr + ", " + Arrays.toString(holes) + ", " + Arrays.toString(holeIds) 
  }
}

