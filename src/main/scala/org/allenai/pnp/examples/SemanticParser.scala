package org.allenai.pnp.examples

import scala.collection.JavaConverters._
import scala.collection.mutable.HashMap
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.{ Map => MutableMap }
import scala.collection.mutable.{ Set => MutableSet }
import scala.collection.mutable.MultiMap

import org.allenai.pnp.Pp
import org.allenai.pnp.Pp._
import org.allenai.pnp.PpUtil

import com.jayantkrish.jklol.ccg.lambda.Type
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.ccg.lambda2.StaticAnalysis
import com.jayantkrish.jklol.ccg.lambda2.ExpressionSimplifier
import scala.collection.mutable.Queue
import com.google.common.base.Preconditions
import java.util.Arrays

case class ExpressionPart(val expr: Expression2,
    val holes: Array[Int], val holeIds: Array[Int]) {
  Preconditions.checkArgument(holes.length == holeIds.length)
  
  override def toString: String = {
    "ExpressionPart(" + expr + ", " + Arrays.toString(holes) + ", " + Arrays.toString(holeIds) 
  }
}

case class SemanticParserState(val parts: Map[Int, ExpressionPart],
    val unfilledHoleIds: List[(Int, Type, Scope)], val nextId: Int) {
  
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
}

object SemanticParserState {
  def start(t: Type): SemanticParserState = {
    val scope = Scope(List.empty)    
    SemanticParserState(Map.empty, List((0, t, scope)), 1) 
  }
}

sealed trait Template {
  val root: Type
  val holeIndexes: Array[Int]
  
  def apply(state: SemanticParserState): SemanticParserState
  
  def matches(expIndex: Int, exp: Expression2, typeMap: Map[Integer, Type]): Boolean
}

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

class Lexicon(
    val typeTemplateMap: MultiMap[Type, Template],
    val rootTypes: List[Type]
    ) {

  def getTemplates(t: Type): Vector[Template] = {
    typeTemplateMap.getOrElse(t, Set.empty).toVector
  }
}

case class ExpressionLabel(val expr: Expression2, val typeMap: Map[Integer, Type])

object ExpressionLabel {
  def fromExpression(expr: Expression2, typeDeclaration: TypeDeclaration): ExpressionLabel = {
    val canonicalExpr = ExpressionSimplifier.lambdaCalculus().apply(expr)
    
    val typeMap = StaticAnalysis.inferTypeMap(canonicalExpr, TypeDeclaration.TOP, typeDeclaration).asScala
    ExpressionLabel(canonicalExpr, typeMap.toMap)
  }
}

case class Scope(val vars: List[(Expression2, Type)]) {

  def getVariableExpressions(t: Type): List[Expression2] = {
    vars.filter(_._2.equals(t)).map(_._1)
  }
  
  def getVariableTemplates(t: Type): List[Template] = {
    getVariableExpressions(t).map(x => ConstantTemplate(t, x))
  }
  
  def extend(types: List[Type]): (Scope, List[String]) = {
    var varNames = List[String]()
    var nextVars = vars
    for (t <- types) {
      val varName = "$" + nextVars.size
      varNames = varName :: varNames
      nextVars = (Expression2.constant(varName), t) :: nextVars
    }

    val nextScope = new Scope(nextVars)
    (nextScope, varNames)
  }
}

class SemanticParser(lexicon: Lexicon) {

  def generateExpression(): Pp[Expression2] = {
    for {
      rootType <- choose(lexicon.rootTypes)
      e <- generateExpression(SemanticParserState.start(rootType)) 
    } yield {
      e
    }
  }
  
  def generateExpression(rootType: Type): Pp[Expression2] = {
    generateExpression(SemanticParserState.start(rootType))
  }

  def generateExpression(state: SemanticParserState): Pp[Expression2] = {
    if (state.unfilledHoleIds.length == 0) {
      Pp.value(state.decodeExpression)
    } else {
      val (holeId, t, scope) = state.unfilledHoleIds.head
      val applicableTemplates = lexicon.getTemplates(t) ++ scope.getVariableTemplates(t)

      // TODO: variables from lambda expressions
      for {
        template <- choose(applicableTemplates)
        nextState = template.apply(state)
        expr <- generateExpression(nextState)
      } yield {
        expr
      }
    }
  }
  
  def generateActionSequence(exp: Expression2, typeDeclaration: TypeDeclaration): List[Template] = {
    val indexQueue = ListBuffer[(Int, Scope)]()
    indexQueue += ((0, Scope(List.empty))) 
    val actions = ListBuffer[Template]()

    val typeMap = StaticAnalysis.inferTypeMap(exp, TypeDeclaration.TOP, typeDeclaration).asScala.toMap
    var state = SemanticParserState.start(typeMap(0))
    
    while (indexQueue.size > 0) {
      val (expIndex, currentScope) = indexQueue.head
      indexQueue.remove(0)

      val curType = typeMap(expIndex)
      val templates = lexicon.typeTemplateMap(curType) ++ currentScope.getVariableTemplates(curType)

      val matches = templates.filter(_.matches(expIndex, exp, typeMap))
      Preconditions.checkState(matches.size == 1, "Found %s matches for expression %s (expected 1)",
          matches.size.asInstanceOf[AnyRef], exp.getSubexpression(expIndex))
      val theMatch = matches.toList(0)
      state = theMatch.apply(state)
      val nextScopes = state.unfilledHoleIds.takeRight(theMatch.holeIndexes.size).map(x => x._3)
      
      actions += theMatch
      var holeOffset = 0
      for ((holeIndex, nextScope) <- theMatch.holeIndexes.zip(nextScopes)) {
        val curIndex = expIndex + holeIndex + holeOffset
        indexQueue += ((curIndex, nextScope))
        holeOffset += (exp.getSubexpression(curIndex).size - 1)
        // println("   Queued: " + curIndex + " " +  exp.getSubexpression(curIndex))
      }

      // println(theMatch)
      // println(indexQueue)
    }
    
    val decoded = state.decodeExpression
    Preconditions.checkState(decoded.equals(exp), "Expected %s and %s to be equal", decoded, exp)
    
    actions.toList
  }
}

object SemanticParser {
  
  def generateLexicon(data: Seq[Expression2], typeDeclaration: TypeDeclaration): Lexicon = {
    val applicationTemplates = for {
      x <- data
      template <- SemanticParser.generateApplicationTemplates(x, typeDeclaration) 
    } yield {
      template
    }
  
    val lambdaTemplates = for {
      x <- data
      template <- SemanticParser.generateLambdaTemplates(x, typeDeclaration) 
    } yield {
      template
    }
  
    val constantTemplates = for {
      x <- data
      typeMap = StaticAnalysis.inferTypeMap(x, TypeDeclaration.TOP, typeDeclaration).asScala
      constant <- StaticAnalysis.getFreeVariables(x).asScala
      typeInd <- StaticAnalysis.getIndexesOfFreeVariable(x, constant)
      t = typeMap(typeInd)
    } yield {
      ConstantTemplate(t, Expression2.constant(constant))
    }
    
    val rootTypes = for {
      x <- data
      typeMap = StaticAnalysis.inferTypeMap(x, TypeDeclaration.TOP, typeDeclaration).asScala
    } yield {
      typeMap(0)
    }
  
    val allTemplates = (applicationTemplates ++ lambdaTemplates ++ constantTemplates)
    val templateMap = allTemplates.map(x => (x.root, x))

    new Lexicon(seqToMultimap(templateMap), rootTypes.toSet.toList)
  }

  def seqToMultimap[A, B](s: Seq[(A, B)]) = { 
    s.foldLeft(new HashMap[A, MutableSet[B]] with MultiMap[A, B]){ 
      (acc, pair) => acc.addBinding(pair._1, pair._2)
    }
  }

  def generateLambdaTemplates(
      e: Expression2,
      typeDeclaration: TypeDeclaration
    ): List[LambdaTemplate] = {
    val typeMap = StaticAnalysis.inferTypeMap(e, TypeDeclaration.TOP, typeDeclaration).asScala
    val builder = ListBuffer[LambdaTemplate]()

    for (scope <- StaticAnalysis.getScopes(e).getScopes.asScala) {
      if (scope.getStart != 0) {
        val i = scope.getStart - 1

        val root = typeMap(i)
        val argTypes = StaticAnalysis.getLambdaArgumentIndexes(e, i).map(typeMap(_)).toList
        val bodyType = typeMap(StaticAnalysis.getLambdaBodyIndex(e, i))
      
        builder += LambdaTemplate(root, argTypes, bodyType)
      }
    }

    builder.toList
  }
  
  def generateApplicationTemplates(
      e: Expression2,
      typeDeclaration: TypeDeclaration
    ): List[ApplicationTemplate] = {
    val typeMap = StaticAnalysis.inferTypeMap(e, TypeDeclaration.TOP, typeDeclaration)
    val builder = ListBuffer[ApplicationTemplate]()
    generateApplicationTemplates(e, 0, typeMap.asScala, builder)
    builder.toList
  }
  
  def generateApplicationTemplates(
      e: Expression2,
      index: Int,
      typeMap: MutableMap[Integer, Type],
      builder: ListBuffer[ApplicationTemplate]
    ): Unit = {
    if (StaticAnalysis.isLambda(e, index)) {
      generateApplicationTemplates(e, StaticAnalysis.getLambdaBodyIndex(e, index),
          typeMap, builder)
    } else {
      val subexpr = e.getSubexpression(index)
      if (!subexpr.isConstant) {
        val rootType = typeMap(index)
        val subtypes = e.getChildIndexes(index).map(x => typeMap(x)).toList
        builder += ApplicationTemplate(rootType, subtypes)
      
        for (childIndex <- e.getChildIndexes(index)) {
          generateApplicationTemplates(e, childIndex, typeMap, builder)
        }
      }
    }
  }
}