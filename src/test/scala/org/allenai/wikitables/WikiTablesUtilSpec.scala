package org.allenai.wikitables

import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import com.jayantkrish.jklol.ccg.lambda2.{Expression2, ExpressionSimplifier}
import edu.stanford.nlp.sempre.Formula
import org.scalatest.FlatSpec
import edu.stanford.nlp.sempre.Formula

class WikiTablesUtilSpec extends FlatSpec {

  "WikiTablesUtil" should "get the right Sempre lf from PNP lf" in {
    val pnpExpression = ExpressionParser.expression2().parse("(lambda (y) (lambda (x) (+ x y)))")
    val simplifier = ExpressionSimplifier.lambdaCalculus()
    val pnpExpressionSimplified = simplifier.apply(pnpExpression)
    val sempreExpression = WikiTablesUtil.toSempreLogicalForm(pnpExpressionSimplified)
    val expectedSempreExpression = "(lambda x (lambda y (+ (var y) (var x))))"
    assert(sempreExpression == expectedSempreExpression)
  }
  
  it should "convert expressions with lambdas in applications" in {
    val e = "(argmax (number 1) (number 1) (fb:cell.cell.part (!= fb:part.gamecube)) (reverse (lambda x (count (fb:row.row.computer (var x))))))"
    val sempreLf = Formula.fromString(e)
    val converted = WikiTablesUtil.toPnpLogicalForm(sempreLf)
    val expected = "(argmax (number 1) (number 1) (fb:cell.cell.part (!= fb:part.gamecube)) (reverse (lambda ($0) (count (fb:row.row.computer $0)))))"
    
    assert(converted == expected)
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
