package org.allenai.wikitables

import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import com.jayantkrish.jklol.ccg.lambda2.{Expression2, ExpressionSimplifier}
import edu.stanford.nlp.sempre.Formula
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

  it should "replace ! with reverse while converting to PNP lf" in {
    val formula = Formula.fromString("(!fb:row.row.score (fb:row.row.opponent (or fb:cell.vs_bc_lions fb:cell.at_bc_lions)))")
    val expectedPnpLf = "((reverse fb:row.row.score) (fb:row.row.opponent (or fb:cell.vs_bc_lions fb:cell.at_bc_lions)))"
    assert(expectedPnpLf == WikiTablesUtil.toPnpLogicalForm(formula))
  }

  it should "not replace ! with reverse in != while converting to PNP lf" in {
    val formula = Formula.fromString("(!= fb:cell.ulm)")
    val expectedPnpLf = "(!= fb:cell.ulm)"
    assert(expectedPnpLf == WikiTablesUtil.toPnpLogicalForm(formula))
  }

}
