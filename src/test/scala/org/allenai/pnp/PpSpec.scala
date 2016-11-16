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

/** Test cases for the probabilistic programming monad.
  */
class PpSpec extends FlatSpec with Matchers {

  val TOLERANCE = 0.0001

  "Pp" should "correctly perform inference" in {
    val foo = Pp.chooseMap(Seq((1, 1.0), (2, 2.0)))

    val values = foo.beamSearch(2)
    values.length should be(2)
    values(0) should be((2, 2.0))
    values(1) should be((1, 1.0))
  }

  it should "correctly perform inference (2)" in {
    val foo = for (
      x <- Pp.chooseMap(Seq((1, 1.0), (2, 2.0)));
      y = x + 1;
      z = x + 1
    ) yield (y)

    val values = foo.beamSearch(2)
    values.length should be(2)
    values(0) should be((3, 2.0))
    values(1) should be((2, 1.0))
  }

  it should "correctly perform inference (2.5)" in {
    val foo = for (
      x <- Pp.chooseMap(Seq((1, 1.0), (2, 2.0)));
      y <- Pp.chooseMap(Seq((1, 1.0), (2, 2.0)));
      z <- Pp.chooseMap(Seq((1, 1.0), (2, 2.0)))
    ) yield (x + y + z)

    val values = foo.beamSearch(10)
    values.length should be(8)
    values(0)._1 should be(6)
    values(0)._2 should be(8.0 +- TOLERANCE)
  }

  it should "correctly perform inference (3)" in {
    def foo(k: Int): Pp[List[Boolean]] = {
      if (k == 0) {
        Pp.value(List.empty[Boolean])
      } else {
        for (
          x <- Pp.chooseMap(Seq((true, 2.0), (false, 1.0)));
          y <- foo(k - 1)
        ) yield (x :: y)
      }
    }

    val values = foo(2).beamSearch(100)
    values.length should be(4)
    values(0)._1 should be(List(true, true))
    values(0)._2 should be(4.0 +- TOLERANCE)
    values(3)._1 should be(List(false, false))
    values(3)._2 should be(1.0 +- TOLERANCE)
  }

  it should "correctly perform inference (4)" in {
    def foo(k: Int): Pp[List[Int]] = {
      if (k == 0) {
        Pp.value(List.empty[Int])
      } else {
        for (
          x <- Pp.chooseMap(Seq((k, 1.0), (k + 1, 2.0)));
          y <- foo(k - 1)
        ) yield (x :: y)
      }
    }

    val values = foo(100).beamSearch(100)
    values.length should be(100)
  }

  it should "correctly collapse inference" in {
    def foo(k: Int): Pp[List[Int]] = {
      if (k == 0) {
        Pp.value(List.empty[Int])
      } else {
        for (
          x <- Pp.chooseMap(Seq((k, 1.0), (k + 1, 2.0)));
          y <- foo(k - 1)
        ) yield (x :: y)
      }
    }

    val marginals = foo(3).beamSearch(100, Env.init)
    marginals.searchSteps should be(3)

    val marginalsOneStep = foo(3).inOneStep().beamSearch(100, Env.init)
    marginalsOneStep.searchSteps should be(1)
  }

  it should "correctly collapse inference (2)" in {
    def twoFlips(): Pp[Int] = {
      val pp = for {
        x <- Pp.chooseMap(Seq((0, 1.0), (1, 2.0)))
        y <- Pp.chooseMap(Seq((0, 1.0), (1, 2.0)))
      } yield {
        x + y
      }

      pp.inOneStep()
    }

    def foo(k: Int): Pp[List[Int]] = {
      if (k == 0) {
        Pp.value(List.empty[Int])
      } else {
        for (
          x <- twoFlips();
          y <- foo(k - 1)
        ) yield (x :: y)
      }
    }

    val marginals = foo(3).beamSearch(100, Env.init)
    marginals.searchSteps should be(3)
  }

  it should "work for infinite lists" in {
    def lm(label: List[String]): Pp[List[String]] = {
      val vocab = Seq(("the", 0.5), ("man", 0.25), ("jumped", 0.125), ("<end>", 0.125))
      for (
        x <- Pp.chooseMap(vocab);
        // TODO: scoring model here.
        rest <- if (label == null || x.equals(label.head)) {
          if (!x.equals("<end>")) {
            lm(label.tail)
          } else {
            Pp.value(List.empty[String])
          }
        } else {
          Pp.fail
        };

        result = x :: rest
      ) yield (result)
    }

    val values = lm(List("the", "man", "<end>")).beamSearch(10)
    values.length should be(1)
    values(0)._1 should be(List("the", "man", "<end>"))
    values(0)._2 should be(0.5 * 0.25 * 0.125 +- TOLERANCE)
  }

  it should "work with mutable state" in {
    def foo(k: Int): Pp[Int] = {
      if (k <= 0) {
        Pp.value(0)
      } else {
        for {
          draw <- Pp.chooseMap(Seq((true, 2.0), (false, 1.0)));
          _ <- if (draw) {
            Pp.setVar("foo", k.asInstanceOf[AnyRef])
          } else {
            Pp.value(())
          };

          recurse <- foo(k - 1);
          v <- Pp.getVar[Int]("foo")
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
    def foo(k: Int): Pp[List[Int]] = {
      if (k == 0) {
        Pp.value(List.empty[Int])
      } else {
        for {
          flip <- Pp.param("flip")
          x <- Pp.choose(Array(k, k + 1), flip)
          y <- foo(k - 1)
        } yield (x :: y)
      }
    }

    val paramNames = IndexedList.create(List("flip").asJava)
    val params = new DenseTensor(Array(0), Array(2), Array(1.0, 2.0))

    val model = new PpModel(paramNames, List(params))
    val env = Env.init
    val cg = model.getInitialComputationGraph()

    val values = foo(1).beamSearch(100, env, cg).executions
    values.length should be(2)
    values(0).value should be(List(2))
    val labels = values(0).env.labels
    labels.length should be(1)
    labels(0).getByDimKey(0) should be(0.0)
    labels(0).getByDimKey(1) should be(1.0)

    val values2 = foo(2).beamSearch(100, env, cg).executions
    values2.length should be(4)
    values2(0).value should be(List(3, 2))
    val labels2 = values2(0).env.labels
    labels2.length should be(2)
    labels2(0).getByDimKey(0) should be(0.0)
    labels2(0).getByDimKey(1) should be(1.0)
    labels2(1).getByDimKey(0) should be(0.0)
    labels2(1).getByDimKey(1) should be(1.0)
  }

  it should "be trainable" in {
    def foo(k: Int, label: List[Int]): Pp[List[Int]] = {
      if (k == 0) {
        Pp.value(List.empty[Int])
      } else {
        for (
          flip <- Pp.param("flip");
          x <- Pp.choose(Array(0, 1), flip);

          y <- if (label == null) {
            foo(k - 1, null)
          } else {
            if (x == label.head) {
              foo(k - 1, label.tail)
            } else {
              Pp.fail
            }
          }

        ) yield (x :: y)
      }
    }

    val paramNames = IndexedList.create(List("flip").asJava)
    val v = DiscreteVariable.sequence("foo", 2);
    val family = new ParametricPpModel(paramNames, List(VariableNumMap.singleton(0, "foo", v)));

    val examples = List(
      PpExample.fromDistributions(foo(2, null), foo(2, List(1, 0))),
      PpExample.fromDistributions(foo(3, null), foo(3, List(1, 1, 1)))
    )

    val oracle = new PpLoglikelihoodOracle[List[Int]](100, family)

    val trainer = new Lbfgs(10, 10, 0.0, new NullLogFunction())
    val params = family.getNewSufficientStatistics
    val trainedParams = trainer.train(oracle, params, examples.asJava)

    val model = family.getModelFromParameters(trainedParams)

    val env = Env.init
    val marginals = foo(1, null).beamSearch(100, env, model.getInitialComputationGraph)
    val values = marginals.executions
    val partitionFunction = marginals.partitionFunction
    values.length should be(2)
    values(0).value should be(List(1))
    (values(0).prob / partitionFunction) should be(0.8 +- TOLERANCE)
  }

  it should "learn xor" in {
    def xor(left: Boolean, right: Boolean): Pp[Boolean] = {
      // Build a feature vector from the inputs
      val values = Array.ofDim[Double](2)
      values(0) = if (left) { 1 } else { 0 }
      values(1) = if (right) { 1 } else { 0 }
      val featureVector = new DenseTensor(Array[Int](2), Array[Int](2), values)

      for {
        // Build a 2 layer neural network with a tanh
        // nonlinearity.
        params <- Pp.param("params")
        bias <- Pp.param("bias")
        hidden = ((params * featureVector) + bias).tanh
        params2 <- Pp.param("params2")
        bias2 <- Pp.param("bias2")
        dist = (params2 * hidden) + bias2

        // Choose the output nondeterministically according to 
        // the per-class scores generated by the neural network.
        y <- Pp.choose(Array(false, true), dist)

        // Extraneous values in the computation graph
        // shouldn't cause problems.
        foo = dist * dist
      } yield {
        y
      }
    }

    // Initialize neural net parameters and their dimensionalities.
    val v = DiscreteVariable.sequence("boolean", 2);
    val h = DiscreteVariable.sequence("hidden", 8);
    val inputVar = VariableNumMap.singleton(2, "input", v)
    val hiddenVar = VariableNumMap.singleton(1, "hidden", h)
    val outputVar = VariableNumMap.singleton(0, "output", v)
    val paramNames = IndexedList.create(
      List("params", "bias", "params2", "bias2").asJava
    )

    val family = new ParametricPpModel(
      paramNames,
      List(inputVar.union(hiddenVar), hiddenVar,
        hiddenVar.union(outputVar), outputVar)
    );

    // Create training data.
    val data = List(
      (true, true, false),
      (true, false, true),
      (false, true, true),
      (false, false, false)
    )
    val examples = data.map(x => {
      val unconditional = xor(x._1, x._2)
      val conditional = for {
        y <- unconditional;
        x <- Pp.require(y == x._3)
      } yield {
        y
      }
      PpExample.fromDistributions(unconditional, conditional)
    })

    // Train model
    val oracle = new PpLoglikelihoodOracle[Boolean](100, family)
    val trainer = StochasticGradientTrainer.createWithL2Regularization(
      1000,
      1, 1.0, false, false, 10.0, 0.0, new NullLogFunction()
    )
    val params = family.getNewSufficientStatistics
    params.perturb(1.0)
    val trainedParams = trainer.train(oracle, params, examples.asJava)
    val model = family.getModelFromParameters(trainedParams)

    // Check that training error is zero.
    for (ex <- data) {
      val env = Env.init
      val marginals = xor(ex._1, ex._2).beamSearch(100, env, model.getInitialComputationGraph)
      val values = marginals.executions
      val partitionFunction = marginals.partitionFunction

      values(0).value should be(ex._3)
      (values(0).prob / partitionFunction) should be(1.0 +- 0.1)
    }
  }
}
