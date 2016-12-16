package org.allenai.pnp.semparse

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

import com.jayantkrish.jklol.ccg.lambda.Type
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.ccg.lambda2.StaticAnalysis

/** Template represents an action that expands a type
  * to an expression. The expression itself may contain
  * typed "holes" to be filled by future templates. 
  * Applying a template to a semantic parser state
  * expands the current hole in the state with the template,
  * returning the next state. 
  */
sealed trait Template {
  val root: Type
  val holeIndexes: Array[Int]
  
  /** Update state by applying this template to expand the  
    * current hole.
    */
  def apply(state: SemanticParserState): SemanticParserState

  /** Returns true if the subexpression at expIndex of exp could
    * be generated by using this template.
    */
  def matches(expIndex: Int, exp: Expression2, typeMap: Map[Integer, Type]): Boolean
}

/** A function application template that rewrites a type
  * as a function type applied to one or more argument types,
  * e.g., t -> (<e,t> <e>)
  */
case class ApplicationTemplate(val root: Type, val elts: List[Type]) extends Template {
  val varNames: ListBuffer[Expression2] = ListBuffer.empty
  val holesBuffer: ListBuffer[Int] = ListBuffer.empty
  for (i <- 1 to elts.length) {
    varNames += Expression2.constant("$$" + i)
    holesBuffer += i
  }

  val expr = Expression2.nested(varNames.toList.asJava)
  val holeIndexes = holesBuffer.toArray
  val holeTypes = elts.toArray
  
  override def apply(state: SemanticParserState): SemanticParserState = {
    val holeIds = ListBuffer.empty[Int]
    for (i <- state.nextId until (state.nextId + holeIndexes.length)) {
      holeIds += i
    }

    val filled = state.unfilledHoleIds.head
    val holeScope = filled._3

    val part = (filled._1, ExpressionPart(expr, holeIndexes.toArray, holeIds.toArray))

    val nextHoles = holeIds.zip(holeTypes).map(x => (x._1, x._2, holeScope))
    val next = state.unfilledHoleIds.drop(1) ++ nextHoles  

    SemanticParserState(state.parts + part, next, state.nextId + holeIndexes.length)
  }

  override def matches(expIndex: Int, exp: Expression2, typeMap: Map[Integer, Type]): Boolean = {
    val subexp = exp.getSubexpression(expIndex)
    if (!subexp.isConstant) {
      val childIndexes = exp.getChildIndexes(expIndex).toList
      childIndexes.map(typeMap(_)).equals(elts)
    } else {
      false
    }
  }
}

/** A template generating a constant, e.g., argmax:<<e,t>,t>.
  * This template is the base case of expression generation as
  * it has no holes.
  */
case class ConstantTemplate(val root: Type, val expr: Expression2) extends Template {
  val holeIndexes = Array[Int]()
  
  override def apply(state: SemanticParserState): SemanticParserState = {
    val filled = state.unfilledHoleIds.head
    val part = (filled._1, ExpressionPart(expr, Array.empty[Int], Array.empty[Int]))
    val next = state.unfilledHoleIds.drop(1)
    SemanticParserState(state.parts + part, next, state.nextId + 1)
  }
  
  override def matches(expIndex: Int, exp: Expression2, typeMap: Map[Integer, Type]): Boolean = {
    exp.getSubexpression(expIndex).equals(expr)
  }
}

/** A template that generates a lambda expression.
  */ 
case class LambdaTemplate(val root: Type, val args: List[Type], val body: Type) extends Template {
  val holeIndexes = Array[Int](3 + args.length)
  
  override def apply(state: SemanticParserState): SemanticParserState = {
    val filled = state.unfilledHoleIds.head
    val currentScope = filled._3
    val (nextScope, varNames) = currentScope.extend(args)
    
    val expr = Expression2.lambda(varNames.asJava, Expression2.constant("TEMP"))
    
    val hole = StaticAnalysis.getLambdaBodyIndex(expr, 0)
    val holeId = state.nextId

    val part = (filled._1, ExpressionPart(expr, Array(hole), Array(holeId)))
    
    val next = state.unfilledHoleIds.drop(1) ++ Array((holeId, body, nextScope))
    SemanticParserState(state.parts + part, next, state.nextId + 1)
  }
  
  override def matches(expIndex: Int, exp: Expression2, typeMap: Map[Integer, Type]): Boolean = {
    if (StaticAnalysis.isLambda(exp, expIndex)) {
      val subexpArgIndexes = StaticAnalysis.getLambdaArgumentIndexes(exp, expIndex).toList
      val subexpBodyIndex = StaticAnalysis.getLambdaBodyIndex(exp, expIndex)

      subexpArgIndexes.map(typeMap(_)).equals(args) && typeMap(subexpBodyIndex).equals(body)
    } else {
      false
    }
  }
}