package org.allenai.wikitables

import java.util.regex.Pattern
import scala.collection.JavaConverters._

import edu.stanford.nlp.sempre.LanguageAnalyzer
import edu.stanford.nlp.sempre.tables.TableKnowledgeGraph
import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import com.jayantkrish.jklol.ccg.lambda2.{Expression2, ExpressionSimplifier}
import edu.stanford.nlp.sempre.Formula
import edu.stanford.nlp.sempre.Formula
import edu.stanford.nlp.sempre.tables.test.CustomExample
import fig.basic.LispTree
import org.scalatest._

class WikiTablesUtilSpec extends FlatSpecLike with Matchers {

  def runLogicalFormTest(sempre: String, pnp: String): Unit = {
    val pnpExpression = ExpressionParser.expression2().parse(pnp)
    val sempreFormula = Formula.fromString(sempre)

    val pnpToSempre = WikiTablesUtil.toSempreLogicalForm(pnpExpression)
    assert(pnpToSempre.toString == sempre)

    val sempreToPnp = WikiTablesUtil.toPnpLogicalForm(sempreFormula)
    assert(sempreToPnp.toString == pnp)
  }

  "to*LogicalForm" should "get the right Sempre lf from PNP lf" in {
    runLogicalFormTest("(lambda x (lambda y (+ (var x) (var y))))", "(lambda (x) (lambda (y) (+ x y)))")
  }

  it should "convert expressions with lambdas in applications" in {
    runLogicalFormTest("(argmax (number 1) (number 1) (fb:cell.cell.part (!= fb:part.gamecube)) (reverse (lambda x (count (fb:row.row.computer (var x))))))",
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

  "convertCustomExampleToWikiTablesExample" should "convert all necessary fields" in {
    val utterance = "(utterance \"how many finished all 225 laps?\")"
    val context = "(context (graph tables.TableKnowledgeGraph csv/204-csv/946.csv))"
    val target = "(targetValue (list (description \"8\")))"
    val lispString = s"(example (id nt-39) $utterance $context $target)"
    TableKnowledgeGraph.opts.baseCSVDir = "data/WikiTableQuestions";
    LanguageAnalyzer.opts.languageAnalyzer = "corenlp.CoreNLPAnalyzer";
    CustomExample.opts.allowNoAnnotation = true
    val tree = LispTree.proto.parseFromString(lispString)
    val customExample = CustomExample.fromLispTree(tree, "39")
    customExample.preprocess()
    val wikitablesExample = WikiTablesUtil.convertCustomExampleToWikiTablesExample(customExample)
    wikitablesExample.sentence.getWords.asScala should be(Seq("how", "many", "finished", "all", "225", "laps", "?"))
    wikitablesExample.sentence.getPosTags.asScala should be(Seq("WRB", "JJ", "VBD", "DT", "CD", "NNS", "."))
  }
}
