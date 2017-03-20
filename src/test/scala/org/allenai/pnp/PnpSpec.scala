package org.allenai.pnp

import scala.collection.JavaConverters._


import org.scalatest._

import com.jayantkrish.jklol.models.DiscreteVariable
import com.jayantkrish.jklol.models.VariableNumMap
import com.jayantkrish.jklol.tensor.DenseTensor
import com.jayantkrish.jklol.training.Lbfgs
import com.jayantkrish.jklol.training.NullLogFunction
import com.jayantkrish.jklol.training.StochasticGradientTrainer
import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.dynet._
import scala.collection.mutable.ListBuffer

/** Test cases for the probabilistic programming monad.
  */
class PnpSpec extends FlatSpec with Matchers {

  Initialize.initialize()

  val TOLERANCE = 0.01

  "Pnp" should "perform inference on choices" in {
    val foo = Pnp.chooseMap(Seq((1, 1.0), (2, 2.0)))

    val values = foo.beamSearch(2).executions.map(x => (x.value, x.prob))
    values.length should be(2)
    values(0) should be((2, 2.0))
    values(1) should be((1, 1.0))
    
    
    val numExecutions = 10000
    val executions = for {
      i <- 0 until numExecutions
    } yield {
      foo.sample()
    }
    
    val oneProb = executions.filter(_.value == 1).length.toDouble / numExecutions
    val twoProb = executions.filter(_.value == 2).length.toDouble / numExecutions
    
    oneProb should be(1.0 / 3 +- TOLERANCE)
    twoProb should be(2.0 / 3 +- TOLERANCE)
  }

  it should "perform inference with successive operations" in {
    val foo = for (
      x <- Pnp.chooseMap(Seq((1, 1.0), (2, 2.0)));
      y = x + 1;
      z = x + 1
    ) yield (y)

    val values = foo.beamSearch(2).executions.map(x => (x.value, x.prob))
    values.length should be(2)
    values(0) should be((3, 2.0))
    values(1) should be((2, 1.0))
  }

  it should "perform inference on multiple choices" in {
    val foo = for (
      x <- Pnp.chooseMap(Seq((1, 1.0), (2, 2.0)));
      y <- Pnp.chooseMap(Seq((1, 1.0), (2, 2.0)));
      z <- Pnp.chooseMap(Seq((1, 1.0), (2, 2.0)))
    ) yield (x + y + z)

    val values = foo.beamSearch(10).executions.map(x => (x.value, x.prob))
    values.length should be(8)
    values(0)._1 should be(6)
    values(0)._2 should be(8.0 +- TOLERANCE)
  }

  it should "perform inference on recursive functions" in {
    def foo(k: Int): Pnp[List[Boolean]] = {
      if (k == 0) {
        Pnp.value(List.empty[Boolean])
      } else {
        for (
          x <- Pnp.chooseMap(Seq((true, 2.0), (false, 1.0)));
          y <- foo(k - 1)
        ) yield (x :: y)
      }
    }

    val values = foo(2).beamSearch(100).executions.map(x => (x.value, x.prob))
    values.length should be(4)
    values(0)._1 should be(List(true, true))
    values(0)._2 should be(4.0 +- TOLERANCE)
    values(3)._1 should be(List(false, false))
    values(3)._2 should be(1.0 +- TOLERANCE)
  }

  it should "truncate the beam during beam search" in {
    def foo(k: Int): Pnp[List[Int]] = {
      if (k == 0) {
        Pnp.value(List.empty[Int])
      } else {
        for (
          x <- Pnp.chooseMap(Seq((k, 1.0), (k + 1, 2.0)));
          y <- foo(k - 1)
        ) yield (x :: y)
      }
    }

    val values = foo(100).beamSearch(100).executions.map(x => (x.value, x.prob))
    values.length should be(100)
  }

  it should "collapse inference" in {
    def foo(k: Int): Pnp[List[Int]] = {
      if (k == 0) {
        Pnp.value(List.empty[Int])
      } else {
        for (
          x <- Pnp.chooseMap(Seq((k, 1.0), (k + 1, 2.0)));
          y <- foo(k - 1)
        ) yield (x :: y)
      }
    }

    val marginals = foo(3).beamSearch(100, Env.init)
    marginals.searchSteps should be(4)

    val marginalsOneStep = foo(3).inOneStep().beamSearch(100, Env.init)
    marginalsOneStep.searchSteps should be(1)
  }

  it should "collapse inference (2)" in {
    def twoFlips(): Pnp[Int] = {
      val pp = for {
        x <- Pnp.chooseMap(Seq((0, 1.0), (1, 2.0)))
        y <- Pnp.chooseMap(Seq((0, 1.0), (1, 2.0)))
      } yield {
        x + y
      }

      pp.inOneStep()
    }

    def foo(k: Int): Pnp[List[Int]] = {
      if (k == 0) {
        Pnp.value(List.empty[Int])
      } else {
        for (
          x <- twoFlips();
          y <- foo(k - 1)
        ) yield (x :: y)
      }
    }

    val marginals = foo(3).beamSearch(100, Env.init)
    marginals.searchSteps should be(4)
  }

  it should "work for infinite lists" in {
    def lm(label: List[String]): Pnp[List[String]] = {
      val vocab = Seq(("the", 0.5), ("man", 0.25), ("jumped", 0.125), ("<end>", 0.125))
      for (
        x <- Pnp.chooseMap(vocab);
        // TODO: scoring model here.
        rest <- if (label == null || x.equals(label.head)) {
          if (!x.equals("<end>")) {
            lm(label.tail)
          } else {
            Pnp.value(List.empty[String])
          }
        } else {
          Pnp.fail
        };

        result = x :: rest
      ) yield (result)
    }

    val values = lm(List("the", "man", "<end>")).beamSearch(10).executions.map(x => (x.value, x.prob))
    values.length should be(1)
    values(0)._1 should be(List("the", "man", "<end>"))
    values(0)._2 should be(0.5 * 0.25 * 0.125 +- TOLERANCE)
  }

  it should "work with mutable state" in {
    def foo(k: Int): Pnp[Int] = {
      if (k <= 0) {
        Pnp.value(0)
      } else {
        for {
          draw <- Pnp.chooseMap(Seq((true, 2.0), (false, 1.0)));
          _ <- if (draw) {
            Pnp.setVar("foo", k.asInstanceOf[AnyRef])
          } else {
            Pnp.value(())
          };

          recurse <- foo(k - 1);
          v <- Pnp.getVar[Int]("foo")
        } yield {
          v
        }
      }
    }

    val env = Env.init.setVar("foo", 4.asInstanceOf[AnyRef])
    val values = foo(3).beamSearch(50, env).executions
    values.length should be(8)

    val mapped = values.map(x => (x.value, x.prob))
    mapped(0)._1 should be(1)
    mapped(0)._2 should be(8.0 +- TOLERANCE)

    values.slice(1, 4).map(x => x.value).sortBy(x => x) should be(Seq(1, 1, 2))
  }

  it should "use parameters and track choices" in {
    def foo(k: Int): Pnp[List[Int]] = {
      if (k == 0) {
        Pnp.value(List.empty[Int])
      } else {
        for {
          flip <- Pnp.param("flip")
          x <- Pnp.choose(Array(k, k + 1), flip)
          _ <- if (x == 2) { Pnp.score(2.0) } else { Pnp.score(1.0) }
          y <- foo(k - 1)
        } yield (x :: y)
      }
    }

    val model = PnpModel.init(false)
    val flipParam = model.addParameter("flip", Dim(2))
    flipParam.zero()
    
    val env = Env.init
    val context = PnpInferenceContext.init(model)

    val values = foo(1).beamSearch(100, env, context).executions
    values.length should be(2)
    values(0).value should be(List(2))
    val labels = values(0).env.labels
    labels.length should be(1)
    labels(0) should be(1)

    val values2 = foo(2).beamSearch(100, env, context).executions
    values2.length should be(4)
    values2(0).value should be(List(2, 2))
    val labels2 = values2(0).env.labels
    labels2.length should be(2)
    labels2(0) should be(1)
    labels2(1) should be(0)
  }
}
