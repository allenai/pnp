package org.allenai.wikitables

import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import com.jayantkrish.jklol.ccg.lambda2.{Expression2, ExpressionSimplifier}
import edu.stanford.nlp.sempre.Formula
import org.scalatest.FlatSpec
import edu.stanford.nlp.sempre.Formula

class WikiTablesUtilSpec extends FlatSpec {

  "WikiTablesUtil" should "get the right Sempre lf from PNP lf" in {
    runTest("(lambda x (lambda y (+ (var x) (var y))))", "(lambda (x) (lambda (y) (+ x y)))")
  }
  
  it should "convert expressions with lambdas in applications" in {
    runTest("(argmax (number 1) (number 1) (fb:cell.cell.part (!= fb:part.gamecube)) (reverse (lambda x (count (fb:row.row.computer (var x))))))",
        "(argmax (number 1) (number 1) (fb:cell.cell.part (!= fb:part.gamecube)) (reverse (lambda (x) (count (fb:row.row.computer x)))))")
  }

  it should "replace ! with reverse while converting to PNP lf" in {
    val formula = Formula.fromString("(!fb:row.row.score (fb:row.row.opponent (or fb:cell.vs_bc_lions fb:cell.at_bc_lions)))")
    val expectedPnpLf = "((reverse fb:row.row.score) (fb:row.row.opponent (or fb:cell.vs_bc_lions fb:cell.at_bc_lions)))"
    assert(expectedPnpLf == WikiTablesUtil.toPnpLogicalForm(formula).toString)
  }

  it should "not replace ! with reverse in != while converting to PNP lf" in {
    val formula = Formula.fromString("(!= fb:cell.ulm)")
    val expectedPnpLf = "(!= fb:cell.ulm)"
    assert(expectedPnpLf == WikiTablesUtil.toPnpLogicalForm(formula).toString)
  }
  
  def runTest(sempre: String, pnp: String): Unit = {
    val pnpExpression = ExpressionParser.expression2().parse(pnp)
    val sempreFormula = Formula.fromString(sempre)

    val pnpToSempre = WikiTablesUtil.toSempreLogicalForm(pnpExpression)
    assert(pnpToSempre.toString == sempre)

    val sempreToPnp = WikiTablesUtil.toPnpLogicalForm(sempreFormula)
    assert(sempreToPnp.toString == pnp)
  }
}
