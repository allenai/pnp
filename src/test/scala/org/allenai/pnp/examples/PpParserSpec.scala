package org.allenai.pnp.examples

import scala.collection.JavaConverters._

import org.allenai.pnp.Env
import org.allenai.pnp.Pp
import org.allenai.pnp.PpLoglikelihoodOracle
import org.allenai.pnp.PpExample
import org.scalatest.FlatSpec
import org.scalatest.Matchers

import com.jayantkrish.jklol.training.Lbfgs
import com.jayantkrish.jklol.training.NullLogFunction

class PpParserSpec extends FlatSpec with Matchers {

  val lexicon = List(
    ("the", "DT"),
    ("the", "NN"),
    ("blue", "NN"),
    ("man", "NN")
  )

  val grammar = List(
    (("DT", "NN"), "NP"),
    (("NN", "NN"), "NN")
  )

  val trainingData = List[Parse](
    Nonterminal(Terminal("the", "DT"), Terminal("man", "NN"), "NP"),
    Nonterminal(Terminal("blue", "NN"), Terminal("man", "NN"), "NN")
  )

  val parser = PpParser.fromMaps(lexicon, grammar)
  val family = parser.getParams
  val model = family.getModelFromParameters(family.getNewSufficientStatistics)

  "PpParser" should "parse sentences" in {
    val dist = parser.parse(List("the", "blue", "man"))

    val env = Env.init
    val cg = model.getInitialComputationGraph
    val marginals = dist.beamSearch(100, env, cg)

    val parses = marginals.executions
    println(parses.length)
    println(parses)
  }

  it should "be trainable" in {
    val examples = trainingData.map(tree => {
      val unconditional = parser.parse(tree.getTokens)
      val conditional = for {
        parse <- unconditional;
        _ <- Pp.require(parse == tree)
      } yield {
        parse
      }
      PpExample.fromDistributions(unconditional, conditional)
    })

    val oracle = new PpLoglikelihoodOracle[Parse](100, family)

    val trainer = new Lbfgs(10, 10, 0.0, new NullLogFunction())
    val params = family.getNewSufficientStatistics
    val trainedParams = trainer.train(oracle, params, examples.asJava)

    val model = family.getModelFromParameters(trainedParams)

    // Check that training error is zero.
    for (ex <- trainingData) {
      val env = Env.init
      val marginals = parser.parse(ex.getTokens).beamSearch(
        100, env, model.getInitialComputationGraph
      )
      val values = marginals.executions
      val partitionFunction = marginals.partitionFunction

      values.foreach(println(_))

      values(0).value should be(ex)
    }
  }
}