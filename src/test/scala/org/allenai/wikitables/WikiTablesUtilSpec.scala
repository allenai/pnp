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
import org.json4s.JsonDSL._

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

  val contextString = "(context (graph tables.TableKnowledgeGraph csv/204-csv/946.csv))"
  val goldLogicalFormString = "(max ((reverse fb:cell.cell.date) ((reverse fb:row.row.year) (fb:row.row.league fb:cell.usl_a_league))))"
  val possibleLogicalFormStrings = Seq(
    "(max ((reverse fb:cell.cell.number) ((reverse fb:row.row.year) (fb:row.row.league fb:cell.usl_a_league))))",
    "(max ((reverse fb:cell.cell.number) ((reverse fb:row.row.date) (fb:row.row.league fb:cell.usl_a_league))))",
    "(max ((reverse fb:cell.cell.date) ((reverse fb:row.row.number) (fb:row.row.league fb:cell.usl_a_league))))"
  )
  val customExample = {
    val utterance = "(utterance \"how many finished all 225 laps?\")"
    val target = "(targetValue (list (description \"8\")))"
    val targetFormula = s"(targetFormula $goldLogicalFormString)"
    val lispString = s"(example (id nt-39) $targetFormula $utterance $contextString $target)"
    TableKnowledgeGraph.opts.baseCSVDir = "data/WikiTableQuestions";
    LanguageAnalyzer.opts.languageAnalyzer = "corenlp.CoreNLPAnalyzer";
    CustomExample.opts.allowNoAnnotation = true
    val tree = LispTree.proto.parseFromString(lispString)
    val example = CustomExample.fromLispTree(tree, "39")
    example.preprocess()
    example.alternativeFormulas = possibleLogicalFormStrings.map(Formula.fromString).asJava
    example
  }

  "convertCustomExampleToWikiTablesExample" should "convert all necessary fields" in {
    val wikitablesExample = WikiTablesUtil.convertCustomExampleToWikiTablesExample(customExample)

    wikitablesExample.sentence.getWords.asScala should be(Seq("how", "many", "finished", "all", "225", "laps", "?"))
    wikitablesExample.sentence.getPosTags.asScala should be(Seq("WRB", "JJ", "VBD", "DT", "CD", "NNS", "."))
    wikitablesExample.sentence.getAnnotation("NER").asInstanceOf[Seq[Seq[String]]] should be(
      Seq(Seq(), Seq(), Seq(), Seq(), Seq("NUMBER", "225.0"), Seq(), Seq()))
    wikitablesExample.tableString should be(contextString)
    wikitablesExample.targetValue should be(customExample.targetValue)
  }

  "exampleToJson" should "construct correct json" in {
    val example = WikiTablesUtil.convertCustomExampleToWikiTablesExample(customExample)
    val expectedJson =
      ("id" -> "nt-39") ~
        ("question" -> example.sentence.getWords.asScala.mkString(" ")) ~
        ("tokens" -> example.sentence.getWords.asScala) ~
        ("posTags" -> example.sentence.getPosTags.asScala) ~
        ("NER" -> example.sentence.getAnnotation("NER").asInstanceOf[Seq[Seq[String]]]) ~
        ("table" -> example.tableString) ~
        ("answer" -> example.targetValue.map(_.toLispTree.toString).getOrElse("")) ~
        ("gold logical form" -> goldLogicalFormString) ~
        ("possible logical forms" -> possibleLogicalFormStrings)
    val actualJson = WikiTablesUtil.exampleToJson(example)
    actualJson should be(expectedJson)
  }

  "exampleFromJson" should "read json correctly" in {
    // We're already testing the to json conversion, so we'll just use it here, and make sure we
    // get the same object back out.
    val example = WikiTablesUtil.convertCustomExampleToWikiTablesExample(customExample)
    val readExample = WikiTablesUtil.exampleFromJson(WikiTablesUtil.exampleToJson(example))

    // AnnotatedSentence doesn't implement equals, so we can't just do `readExample should
    // be(example)`.  We need to check all of the fields individually.
    readExample.goldLogicalForm should be(example.goldLogicalForm)
    readExample.possibleLogicalForms should be(example.possibleLogicalForms)
    readExample.tableString should be(example.tableString)
    readExample.targetValue should be(example.targetValue)
    val readSentence = readExample.sentence
    val sentence = example.sentence
    readSentence.getWords should be(sentence.getWords)
    readSentence.getPosTags should be(sentence.getPosTags)
    readSentence.getAnnotations should be(sentence.getAnnotations)
  }
}
