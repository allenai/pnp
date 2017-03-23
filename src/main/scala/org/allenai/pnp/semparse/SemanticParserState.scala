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
    val unfilledHoleIds: List[Hole], val nextId: Int,
    val numActions: Int, val rootType: Type,
    val templates: List[Template], val attentions: List[Expression]) {

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
        rootType, templates, e :: attentions)
  }
  
  def getTemplates: Array[Template] = {
    templates.reverse.toArray
  }
  
  def getAttentions: Array[Expression] = {
    attentions.reverse.toArray
  }
  
  def nextHole(): Option[Hole] = {
    if (unfilledHoleIds.size > 0) {
      Some(unfilledHoleIds(0))
    } else {
      None
    }
  }

  def fill(hole: Hole, part: ExpressionPart, newHoles: List[Hole], template: Template): SemanticParserState = {
    Preconditions.checkArgument(unfilledHoleIds(0).id == hole.id)
    val partTuple = (hole.id, part)
    
    val unfilledHoles = if (hole.repeated) {
      unfilledHoleIds
    } else {
      unfilledHoleIds.drop(1)
    }
    val nextHoles = newHoles ++ unfilledHoles
    
    SemanticParserState(parts + partTuple, nextHoles, nextId + newHoles.length,
        numActions + 1, rootType, template :: templates, attentions)
  }

  def drop(hole: Hole, template: Template): SemanticParserState = {
    Preconditions.checkArgument(unfilledHoleIds(0).id == hole.id)
    SemanticParserState(parts, unfilledHoleIds.drop(1), nextId,
        numActions + 1, rootType, template :: templates, attentions)
  }

  def hasRootType: Boolean = {
    rootType != null
  }

  def addRootType(rootType: Type): SemanticParserState = {
    Preconditions.checkState(unfilledHoleIds.length == 0 && numActions == 0,
        "The root type can only be added at the beginning of parsing".asInstanceOf[AnyRef])
    
    val scope = Scope(List.empty)
    SemanticParserState(parts, List(Hole(0, rootType, scope, false)), 1, 0, rootType, List(), List())
  }
}

object SemanticParserState {

  /** The start state of a semantic parser. The expected
    * use of this state is to call addRootType, followed by
    * applying a sequence of templates.  
    */
  def start(): SemanticParserState = {
    SemanticParserState(Map.empty, List(), 1, 0, null, List(), List())
  }
}

case class ExpressionPart(val expr: Expression2,
    val holes: Array[Int], val holeIds: Array[Int]) {
  Preconditions.checkArgument(holes.length == holeIds.length)
  
  override def toString: String = {
    "ExpressionPart(" + expr + ", " + Arrays.toString(holes) + ", " + Arrays.toString(holeIds) 
  }
}

case class Hole(id: Int, t: Type, scope: Scope, repeated: Boolean)