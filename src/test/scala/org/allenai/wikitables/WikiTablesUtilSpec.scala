package org.allenai.wikitables

import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import com.jayantkrish.jklol.ccg.lambda2.{Expression2, ExpressionSimplifier}
import org.scalatest.FlatSpec

class WikiTablesUtilSpec extends FlatSpec {

  behavior of "WikiTablesUtilSpec"

  it should "get the right Sempre lf from PNP lf" in {
    val pnpExpression = ExpressionParser.expression2().parse("(lambda (y) (lambda (x) (+ x y)))")
    val simplifier = ExpressionSimplifier.lambdaCalculus()
    val pnpExpressionSimplified = simplifier.apply(pnpExpression)
    val sempreExpression = WikiTablesUtil.toSempreLogicalForm(pnpExpressionSimplified)
    val expectedSempreExpression = "(lambda x (lambda y (+ (var y) (var x))))"
    assert(sempreExpression == expectedSempreExpression)
  }
}
