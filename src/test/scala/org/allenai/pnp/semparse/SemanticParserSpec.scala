package org.allenai.pnp.semparse

import scala.collection.JavaConverters._
import org.allenai.pnp.{Env, Pnp, PnpInferenceContext, PnpModel}

import org.scalatest.FlatSpec
import org.scalatest.Matchers
import com.jayantkrish.jklol.ccg.lambda.ExplicitTypeDeclaration
import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import com.jayantkrish.jklol.training.NullLogFunction
import com.jayantkrish.jklol.util.IndexedList
import edu.cmu.dynet._

class SemanticParserSpec extends FlatSpec with Matchers {
  
  Initialize.initialize()
 
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
  val model = PnpModel.init(true)
  val config = new SemanticParserConfig()
  val parser = SemanticParser.create(lexicon, vocab, config, model)

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
    val oracle = parser.getLabelScore(label, entityLinking, typeDeclaration).get
    val exprs = parser.generateExpression(Array("major", "city").map(vocab.getIndex(_)),
        entityLinking)

    ComputationGraph.renew()
    val context = PnpInferenceContext.init(model).addExecutionScore(oracle)

    val results = exprs.beamSearch(1, -1, Env.init, context).executions
    results.length should be(1)
    results(0).value should equal(label)
  }
  
  it should "condition on multiple expressions" in {
    val label1 = exprParser.parse("(lambda ($0) (and:<t*,t> (city:<e,t> $0) (major:<e,t> $0)))")
    val label2 = exprParser.parse("(lambda ($0) (state:<e,t> $0))")
    val labels = Set(label1, label2)
    val entityLinking = EntityLinking(List())
    val oracle = parser.getMultiLabelScore(labels, entityLinking, typeDeclaration).get
    
    val exprs = parser.generateExpression(Array("major", "city").map(vocab.getIndex(_)),
        entityLinking)

    ComputationGraph.renew()
    val context = PnpInferenceContext.init(model).addExecutionScore(oracle)

    val results = exprs.beamSearch(2, -1, Env.init, context).executions
    results.length should be(2)
    results.map(_.value).toSet should equal(labels)
  }
}
