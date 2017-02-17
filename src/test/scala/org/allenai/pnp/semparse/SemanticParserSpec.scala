package org.allenai.pnp.semparse

import scala.collection.JavaConverters._

import org.allenai.pnp.Env
import org.scalatest.FlatSpec
import org.scalatest.Matchers

import com.jayantkrish.jklol.ccg.lambda.ExplicitTypeDeclaration
import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import com.jayantkrish.jklol.training.NullLogFunction
import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._
import org.allenai.pnp.PpModel

class SemanticParserSpec extends FlatSpec with Matchers {
  
  initialize(new DynetParams())
 
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

  val lexicon = ActionSpace.fromExpressions(data.map(_._2), typeDeclaration, true)
  val vocab = IndexedList.create[String]
  for (d <- data) {
    vocab.addAll(d._1.toList.asJava)
  }
  val model = PpModel.init(true)
  val parser = SemanticParser.create(lexicon, vocab, model)

  "SemanticParser" should "generate application templates" in {
    println(lexicon.typeTemplateMap)
  }

  it should "decode expressions to template sequences" in {
    val e = exprParser.parse(
        "(argmax:<<e,t>,e> (lambda ($0) (and:<t*,t> (city:<e,t> $0) (major:<e,t> $0))))")
    // This method will throw an error if it can't decode the expression properly. 
    val templates = parser.generateActionSequence(e, EntityLinking(List()), typeDeclaration)
  }
  
  it should "condition on expressions" in {
    val label = exprParser.parse("(lambda ($0) (and:<t*,t> (city:<e,t> $0) (major:<e,t> $0)))")
    val entityLinking = EntityLinking(List())
    val oracle = parser.generateExecutionOracle(label, entityLinking, typeDeclaration).get
    val exprs = parser.generateExpression(Array("major", "city").map(vocab.getIndex(_)), entityLinking)

    val cg = new ComputationGraph
    val compGraph = parser.model.getComputationGraph(cg)
    
    val results = exprs.beamSearch(1, -1, Env.init, oracle, compGraph, new NullLogFunction()).executions
    for (result <- results) {
      println("  " + result)
    }

    cg.delete()

    results.length should be(1)
    results(0).value should equal(label)
  }
}
