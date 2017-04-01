package org.allenai.pnp.semparse

import scala.collection.JavaConverters._
import scala.collection.mutable.MultiMap
import scala.collection.mutable.{Map => MutableMap}
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration
import com.jayantkrish.jklol.ccg.lambda2.StaticAnalysis
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import scala.collection.mutable.ListBuffer
import com.jayantkrish.jklol.ccg.lambda.Type
import com.jayantkrish.jklol.util.IndexedList

/** A collection of actions to be used in a semantic parser
 *  to generate expressions. An action space has a set
 *  of root types for the entire expression. It also has,
 *  for each type, a collection of templates for generating
 *  expressions of that type. 
 */
class ActionSpace(
    val typeTemplateMap: MultiMap[Type, Template],
    val rootTypes: Array[Type],
    val allTypes: Set[Type]
    ) extends Serializable {
  
  val typeIndex = IndexedList.create(allTypes.asJava)

  def getTemplates(t: Type): Vector[Template] = {
    typeTemplateMap.getOrElse(t, Set.empty).toVector
  }
  
  def getTypes(): Set[Type] = {
    allTypes
  }
}

object ActionSpace {
  
  /**
   * Create an {@code ActionSpace} from a collection of typed
   * constants. The set of root types consists of all atomic
   * types used in any of the constants' types. The templates
   * permit every function type to be applied to the appropriate
   * arguments. 
   */
  def fromLfConstants(constants: Iterable[String], typeDeclaration: TypeDeclaration
      ): ActionSpace = {
     val typedConstants = constants.map(x => (x, typeDeclaration.getType(x))).toList
     val atomicTypes = typedConstants.map(x => getAtomicTypes(x._2)).flatten

     val applicationTemplates = for {
       (c, t) <- typedConstants.filter(_._2.isFunctional)
       (args, ret) = decomposeType(t)
     } yield {
       ApplicationTemplate.fromTypes(ret, (t :: args).map(x => (x, false)))
     }

     val constantTemplates = for {
       (c, t) <- typedConstants 
     } yield {
       ConstantTemplate(t, Expression2.constant(c))
     }

     // TODO: lambda templates?

     // TODO: maybe allow the root type to be non-atomic
     val rootTypes = atomicTypes
     val allTemplates = applicationTemplates ++ constantTemplates
     val allTypes = rootTypes ++ allTemplates.map(_.root) ++
      applicationTemplates.map(_.holeTypes).flatten

     val templateMap = allTemplates.map(x => (x.root, x))
     new ActionSpace(SemanticParser.seqToMultimap(templateMap),
         rootTypes.toSet.toArray, allTypes.toSet)
  }

  private def getAtomicTypes(t: Type): Set[Type] = {
    if (t.isAtomic()) {
      if (t.hasTypeVariables()) {
        return Set()
      } else {
        return Set(t)
      }
    } else {
      val arg = getAtomicTypes(t.getArgumentType)
      val ret = getAtomicTypes(t.getReturnType)
      arg ++ ret
    }
  }
  
  private def decomposeType(t: Type): (List[Type], Type) = {
    val args = ListBuffer[Type]()
    var x = t
    while (x.isFunctional) {
      args += x.getArgumentType
      x = x.getReturnType
    }
    
    (args.toList.reverse, x)
  }
  
  /** Create a set of templates that can generate all of
   *  the expressions in data.
   */
  def fromExpressions(data: Seq[Expression2], typeDeclaration: TypeDeclaration,
      combineApplications: Boolean): ActionSpace = {

    // Generate each type of action template.
    val applicationTemplates = for {
      x <- data
      template <- generateApplicationTemplates(x, typeDeclaration)
    } yield {
      template
    }

    val lambdaTemplates = for {
      x <- data
      template <- generateLambdaTemplates(x, typeDeclaration) 
    } yield {
      template
    }
  
    val constantTemplates = for {
      x <- data
      typeMap = StaticAnalysis.inferTypeMap(x, TypeDeclaration.TOP, typeDeclaration).asScala
      constant <- StaticAnalysis.getFreeVariables(x).asScala
      typeInd <- StaticAnalysis.getIndexesOfFreeVariable(x, constant)
      t = typeMap(typeInd)
      if !t.hasTypeVariables()
    } yield {
      ConstantTemplate(t, Expression2.constant(constant))
    }
    
    // Get the root type of every logical form
    val rootTypes = for {
      x <- data
      typeMap = StaticAnalysis.inferTypeMap(x, TypeDeclaration.TOP, typeDeclaration).asScala
    } yield {
      typeMap(0)
    }

    val allTemplates = if (combineApplications) {
      // combineApplications merges application templates with their first
      // argument, i.e., the function being invoked. All constants with
      // the appropriate type are substituted into this hole, resulting
      // in new templates. This flag only makes sense if the first argument
      // is always a constant (and not itself a function).

      val constantsWithType = SemanticParser.seqToMultimap(
          constantTemplates.map(x => (x.root, x)).toSeq)
      val newApplicationTemplates = for {
        app <- applicationTemplates
        constant <- constantsWithType(app.holeTypes(0))
        hole = app.holes(0)
        holeIndex = hole._1
        newExpr = app.expr.substitute(holeIndex, constant.expr) 
      } yield {
        ApplicationTemplate(app.root, newExpr, app.holes.drop(1))
      }

      val functionTypes = applicationTemplates.map(x => x.holeTypes(0)).toSet
      val newConstantTemplates = constantTemplates.filter(x => !functionTypes.contains(x.root))
      
      (newApplicationTemplates ++ lambdaTemplates ++ newConstantTemplates) 
    } else {
      (applicationTemplates ++ lambdaTemplates ++ constantTemplates)
    }
    

    val templateMap = allTemplates.map(x => (x.root, x))
    
    val allTypes = rootTypes ++ allTemplates.map(_.root) ++
      applicationTemplates.map(_.holeTypes).flatten ++ lambdaTemplates.map(_.args).flatten

    new ActionSpace(SemanticParser.seqToMultimap(templateMap),
        rootTypes.toSet.toArray, allTypes.toSet)
  }

  private def generateLambdaTemplates(
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
  
  private def generateApplicationTemplates(
      e: Expression2,
      typeDeclaration: TypeDeclaration
    ): List[ApplicationTemplate] = {
    val typeMap = StaticAnalysis.inferTypeMap(e, TypeDeclaration.TOP, typeDeclaration)
    val builder = ListBuffer[ApplicationTemplate]()
    generateApplicationTemplates(e, 0, typeMap.asScala, builder)
    builder.toSet.toList
  }

  private def generateApplicationTemplates(
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
        val subtypes = e.getChildIndexes(index).map(x => (typeMap(x), false)).toList
        builder += ApplicationTemplate.fromTypes(rootType, subtypes)
      
        for (childIndex <- e.getChildIndexes(index)) {
          generateApplicationTemplates(e, childIndex, typeMap, builder)
        }
      }
    }
  }
}
