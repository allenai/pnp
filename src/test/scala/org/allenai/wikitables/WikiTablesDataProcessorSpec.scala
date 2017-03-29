package org.allenai.wikitables

import java.util.HashSet
import scala.collection.JavaConverters._

import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.ccg.lambda2.ExpressionSimplifier
import com.jayantkrish.jklol.nlpannotation.AnnotatedSentence
import edu.stanford.nlp.sempre.tables.TableKnowledgeGraph
import edu.stanford.nlp.sempre.ContextValue
import edu.stanford.nlp.sempre.Formula
import edu.stanford.nlp.sempre.Values
import edu.stanford.nlp.sempre.tables.test.CustomExample
import org.scalatest._
import org.json4s._
import org.json4s.native.JsonMethods.parse
import fig.basic.LispTree

class WikiTablesDataProcessorSpec extends FlatSpecLike with Matchers {
  implicit val formats = DefaultFormats
  val simplifier = ExpressionSimplifier.lambdaCalculus()
  val logicalFormParser = ExpressionParser.expression2();

  /*
   "exampleFromJson" should "construct an example from the input json" in {
     TableKnowledgeGraph.opts.baseCSVDir = "data/WikiTableQuestions"
     val jsonString = """{
       |"question": "What is the answer?",
       |"tokens": ["What", "is", "the", "answer", "?"],
       |"posTags": ["WDT", "VB", "DT", "NN", "."],
       |"NER": [["", ""], ["", ""], ["", ""], ["", ""], ["", ""]],
       |"table": "(context (graph tables.TableKnowledgeGraph csv/203-csv/634.csv))",
       |"answer": "(targetValue (list (description \"3\")))",
       |"gold logical form": "(count (fb:row.row.category fb:cell.",
       |"logical forms": ["(count (fb:row.row.category fb:cell.choice_tv_sidekick))",
       |                  "(count (fb:row.row.population (fb:cell.cell.number (> (number 2000)))))"
       |                 ]
       |}""".stripMargin
     val json = parse(jsonString)
     val question = (json \ "question").extract[String]
     val tokens = (json \ "tokens").extract[List[String]]
     val posTags = (json \ "posTags").extract[List[String]]
     val tableString = (json \ "table").extract[String]
     val answerString = (json \ "answer").extract[String]
     val logicalFormStrings = (json \ "logicalForms").extract[List[String]]
     val sentence = new AnnotatedSentence(tokens.asJava, posTags.asJava)
     val logicalForms = logicalFormStrings
       .map(Formula.fromString)
       .map(WikiTablesUtil.toPnpLogicalForm)
       .map(x => simplifier.apply(logicalFormParser.parse(x)))
     val exampleString = s"(example ${tableString} ${answerString})"
     val contextValue = new ContextValue(LispTree.proto.parseFromString(tableString).child(0))
     val targetValue = Values.fromLispTree(LispTree.proto.parseFromString(answerString).child(0))
     val example = new WikiTablesExample(
       sentence,
       new HashSet[Expression2](logicalForms.asJava),
       contextValue,
       targetValue
     )

  }
  */
}

