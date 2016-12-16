package org.allenai.pnp.semparse

import scala.collection.JavaConverters._
import org.scalatest.FlatSpec
import org.scalatest.Matchers
import com.jayantkrish.jklol.ccg.lambda.ExplicitTypeDeclaration
import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import org.allenai.pnp.examples.SemanticParser

class SemanticParserSpec extends FlatSpec with Matchers {
 
  val dataStrings = List(
      ("state", "state:<e,t>"),
      ("city", "city:<e,t>"),
      ("biggest city", "(argmax:<<e,t>,e> city:<e,t>)"),
      ("texas", "texas:e"),
      ("major city", "(lambda ($0) (and:<t*,t> (city:<e,t> $0) (major:<e,t> $0)))")
  )

  val exprParser = ExpressionParser.expression2()
  val typeDeclaration = ExplicitTypeDeclaration.getDefault()

  val data = dataStrings.map(x => (x._1.split(" "), exprParser.parse(x._2)))

  val lexicon = SemanticParser.generateLexicon(data.map(_._2), typeDeclaration)
  val parser = new SemanticParser(lexicon)

  "SemanticParser" should "generate application templates" in {
    println(lexicon.typeTemplateMap)
  }

  /*
  it should "beam search" in {
    val exprs = parser.generateExpression(Type.parseFrom("e"))
    val results = exprs.beamSearch(100)
    for (result <- results) {
      println("  " + result)
    }
  }
  */
  
  it should "decode expressions to template sequences" in {
    val e = exprParser.parse(
        "(argmax:<<e,t>,e> (lambda ($0) (and:<t*,t> (city:<e,t> $0) (major:<e,t> $0))))")
    // This method will throw an error if it can't decode the expression properly. 
    val templates = parser.generateActionSequence(e, typeDeclaration)
  }
  
  /*
  it should "condition on expressions" in {
    val label = exprParser.parse("(lambda ($0) (and:<t*,t> (city:<e,t> $0) (major:<e,t> $0)))")
    val exprs = parser.generateLabeledExpression(ExpressionLabel.fromExpression(label, typeDeclaration))
    val results = exprs.beamSearch(100)
    for (result <- results) {
      println("  " + result)
    }
    
    results.length should be(1)
    results(0)._1 should equal(label)
  }
  */
}