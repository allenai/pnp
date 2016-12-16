package org.allenai.pnp.semparse

import scala.collection.JavaConverters._
import scala.collection.mutable.HashMap
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.{ Map => MutableMap }
import scala.collection.mutable.MultiMap
import scala.collection.mutable.{ Set => MutableSet }

import org.allenai.pnp.Pp
import org.allenai.pnp.Pp.choose

import com.google.common.base.Preconditions
import com.jayantkrish.jklol.ccg.lambda.Type
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.ccg.lambda2.StaticAnalysis

/** A parser mapping token sequences to a distribution over
  * logical forms.
  */
class SemanticParser(actionSpace: ActionSpace) {

  def generateExpression(): Pp[Expression2] = {
    for {
      rootType <- choose(actionSpace.rootTypes)
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
      val applicableTemplates = actionSpace.getTemplates(t) ++ scope.getVariableTemplates(t)

      for {
        template <- choose(applicableTemplates)
        nextState = template.apply(state)
        expr <- generateExpression(nextState)
      } yield {
        expr
      }
    }
  }

  /** Generate the sequence of parser actions that produces exp.
    * This method assumes that only one such action sequence exists.  
    */
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
      val templates = actionSpace.typeTemplateMap(curType) ++ currentScope.getVariableTemplates(curType)

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
      }
    }

    val decoded = state.decodeExpression
    Preconditions.checkState(decoded.equals(exp), "Expected %s and %s to be equal", decoded, exp)
    
    actions.toList
  }
}

/** A collection of templates and root types for 
  * use in a semantic parser.
  */
class ActionSpace(
    val typeTemplateMap: MultiMap[Type, Template],
    val rootTypes: List[Type]
    ) {

  def getTemplates(t: Type): Vector[Template] = {
    typeTemplateMap.getOrElse(t, Set.empty).toVector
  }
}

object SemanticParser {
  
  def generateActionSpace(data: Seq[Expression2], typeDeclaration: TypeDeclaration): ActionSpace = {
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

    new ActionSpace(seqToMultimap(templateMap), rootTypes.toSet.toList)
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