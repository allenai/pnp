package org.allenai.wikitables

import scala.annotation.migration
import scala.collection.JavaConverters._
import scala.collection.mutable.{Map => MutableMap}

import com.jayantkrish.jklol.ccg.lambda.AbstractTypeDeclaration
import com.jayantkrish.jklol.ccg.lambda.Type
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.ccg.lambda2.StaticAnalysis
import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import com.google.common.base.Preconditions
import scala.util.Try


trait LfPreprocessor extends Serializable {
  def preprocess(lf: Expression2): Expression2
  
  def postprocess(lf: Expression2): Expression2
  
  def preprocess(ex: WikiTablesExample): WikiTablesExample = {
    val goldLf = for {
      lf <- ex.goldLogicalForm
    } yield {
      preprocess(lf)
    }
    val possibleLfs = ex.possibleLogicalForms.map(preprocess(_))
    WikiTablesExample(ex.id, ex.sentence, goldLf, possibleLfs,
        ex.tableString, ex.targetValue)
  }
}

class Seq2TreeTypeDeclaration extends AbstractTypeDeclaration(Map().asJava) {
  import Seq2TreeTypeDeclaration._
  
  val typeCache = MutableMap[Int, Type]()
  
  private def getApplyType(num: Int): Type = {
    if (typeCache.contains(num)) {
      typeCache(num)
    } else {
      val t = if (num == 0) {
        anyType
      } else {
        Type.createFunctional(anyType, getApplyType(num - 1), false)
      }

      typeCache(num) = t
      t
    }
  }

  override def getType(constant: String): Type = {
    if (constant.startsWith(applyPrefix)) {
      val num = constant.substring(applyPrefix.length).toInt
      getApplyType(num)
    } else {
      anyType
    }
  }
}

object Seq2TreeTypeDeclaration {
  val anyType = Type.parseFrom("any")
  val applyPrefix = "apply:"
  val lambda = "lam"
}

class NullLfPreprocessor extends LfPreprocessor {
  def preprocess(lf: Expression2): Expression2 = { lf }
  def postprocess(lf: Expression2): Expression2 = { lf }
}

class Seq2TreePreprocessor extends LfPreprocessor {
  import Seq2TreeTypeDeclaration._

  val constantMap = MutableMap[Int, Expression2]()
  val lambdaConstant = Expression2.constant(lambda)
  val oldLambdaConstant = Expression2.constant(StaticAnalysis.LAMBDA)
  
  private def getApplyConstant(numArgs: Int): Expression2 = {
    if (!constantMap.contains(numArgs)) {
      constantMap(numArgs) = Expression2.constant(applyPrefix + numArgs)
    }
    return constantMap(numArgs)
  }
  
  override def preprocess(lf: Expression2): Expression2 = {
    if (lf.isConstant) {
      if (lf.getConstant.equals(StaticAnalysis.LAMBDA)) {
        lambdaConstant
      } else {
        lf
      }
    } else {
      val preprocessed = lf.getSubexpressions.asScala.map(s => preprocess(s))
      Expression2.nested((List(getApplyConstant(preprocessed.length)) ++ preprocessed).asJava)
    }
  }

  override def postprocess(lf: Expression2): Expression2 = {
    if (lf.isConstant) {
      if (lf.getConstant.equals(lambda)) {
        oldLambdaConstant
      } else {
        lf
      }
    } else {
      val subexpressions = lf.getSubexpressions.asScala.drop(1).map(postprocess(_))
      Expression2.nested(subexpressions.asJava)
    }
  }
}

class Seq2SeqTypeDeclaration extends AbstractTypeDeclaration(Map().asJava) {
  import Seq2SeqTypeDeclaration._

  override def getType(constant: String): Type = {
    if (constant == end) {
      endType
    } else {
      seqType
    }
  }
}

object Seq2SeqTypeDeclaration {
  val endType = Type.createAtomic("s2s")
  val seqType = Type.createFunctional(endType, endType, false)
  val end = "end"
  val endExpr = Expression2.constant("end")
}

class Seq2SeqPreprocessor extends LfPreprocessor {
  import Seq2SeqTypeDeclaration._
  
  val parser = ExpressionParser.expression2()

  override def preprocess(lf: Expression2): Expression2 = {
    val tokens = parser.tokenize(lf.toString).asScala
    
    val replacedTokens = tokens.map(t => t match {
      case "(" => "LRB"
      case ")" => "RRB"
      case StaticAnalysis.LAMBDA => "lam"
      case _ => t
    })

    replacedTokens.foldRight(endExpr)((x, y) => Expression2.nested(Expression2.constant(x), y))
  }
  
  private def expressionToTokens(lf: Expression2): List[String] = {
    if (lf.isConstant()) {
      List()
    } else {
      val subexprs = lf.getSubexpressions.asScala
      Preconditions.checkState(subexprs.length == 2 && subexprs(0).isConstant)
      val token = subexprs(0).getConstant
      token :: expressionToTokens(subexprs(1))
    }
  }

  override def postprocess(lf: Expression2): Expression2 = {
    val tokens = expressionToTokens(lf)
    val replaced = tokens.map(t => t match {
      case "LRB" => "("
      case "RRB" => ")"
      case "lam" => StaticAnalysis.LAMBDA
      case _ => t
    })

    val expr = Try(parser.parse(replaced.asJava))
    expr.getOrElse(endExpr)
  }
}
