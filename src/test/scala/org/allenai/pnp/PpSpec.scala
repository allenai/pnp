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
import edu.cmu.dynet.dynet_swig._
import org.allenai.pnp.examples.DynetScalaHelpers._
import scala.collection.mutable.ListBuffer

/** Test cases for the probabilistic programming monad.
  */
class PpSpec extends FlatSpec with Matchers {

  myInitialize()

  val TOLERANCE = 0.01

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
    marginals.searchSteps should be(4)

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
    marginals.searchSteps should be(4)
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
          _ <- if (x == 2) { Pp.score(2.0) } else { Pp.score(1.0) }
          y <- foo(k - 1)
        } yield (x :: y)
      }
    }

    val computationGraph = new ComputationGraph
    
    val m = new Model
    val paramNames = IndexedList.create[String]
    val flipParam = m.add_parameters(Seq(2))
    paramNames.add("flip")
    val model = new PpModel(paramNames, Array(flipParam),
        IndexedList.create[String], Array(), m, false)
    flipParam.zero()
    
    val env = Env.init
    val cg = model.getInitialComputationGraph(computationGraph)

    val values = foo(1).beamSearch(100, env, cg).executions
    values.length should be(2)
    values(0).value should be(List(2))
    val labels = values(0).env.labels
    labels.length should be(1)
    labels(0) should be(1)

    val values2 = foo(2).beamSearch(100, env, cg).executions
    values2.length should be(4)
    values2(0).value should be(List(2, 2))
    val labels2 = values2(0).env.labels
    labels2.length should be(2)
    labels2(0) should be(1)
    labels2(1) should be(0)
    
    computationGraph.delete()
  }

  it should "be trainable" in {
    def foo(k: Int, label: List[Int]): Pp[List[Int]] = {
      if (k == 0) {
        Pp.value(List.empty[Int])
      } else {
        for (
          flip <- Pp.param("flip");
          x <- Pp.choose(Array(0, 1), flip, k);

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
    
    val m = new Model
    val paramNames = IndexedList.create[String]
    val flipParam = m.add_parameters(Seq(2))
    paramNames.add("flip")
    val model = new PpModel(paramNames, Array(flipParam),
        IndexedList.create[String], Array(), m, true)
    
    val examples = List(
//      PpExample.fromDistributions(foo(2, null), foo(1, List(1))),
        PpExample.fromDistributions(foo(2, null), foo(2, List(1, 0))),
        PpExample.fromDistributions(foo(3, null), foo(3, List(1, 1, 1)))
    )

    val sgd = new SimpleSGDTrainer(model.model, 0.1f, 0.1f)
    val trainer = new LoglikelihoodTrainer(1000, 100, model, sgd, new NullLogFunction())
//  val trainer = new GlobalLoglikelihoodTrainer(1000, 100, model, sgd, new NullLogFunction())
    
    trainer.train(examples)
    
    
    val env = Env.init
    val computationGraph = new ComputationGraph
    val marginals = foo(1, null).beamSearch(100, env, model.getInitialComputationGraph(computationGraph))
    val values = marginals.executions
    val partitionFunction = marginals.partitionFunction
    values.length should be(2)
    values(0).value should be(List(1))
    (values(0).prob / partitionFunction) should be(0.8 +- TOLERANCE)
    
    computationGraph.delete()
  }

  it should "be trainable with global normalization" in {
    val vocab = Array(0,1,2)
    
    def lm(k: Int): Pp[Array[Int]] = {
      if (k == 1) {
        for {
          params <- Pp.param("start")
          choice <- Pp.choose(vocab, params, k - 1)
        } yield {
          Array(choice)
        }
      } else {
        for {
          rest <- lm(k - 1)
          previous = rest.last
          transition <- Pp.param("transition")
          params = pickrange(transition, previous * vocab.length, (previous + 1) * vocab.length)
          choice <- Pp.choose(vocab, params, k - 1)
        } yield {
          rest ++ Array(choice)
        }
      }
    }

    def makeOracle(label: Array[Int]): ExecutionScore = {
      new ExecutionScore() {
        def apply(tag: Any, choice: Any, env: Env): Double = {
          if (tag != null && tag.isInstanceOf[Int]) {
            val tagInt = tag.asInstanceOf[Int]
            if (tagInt >= 0 && tagInt < label.length) {
              if (choice == label(tagInt)) {
                0.0
              } else {
                Double.NegativeInfinity
              }
            } else {
              Double.NegativeInfinity
            }
          } else {
            0.0
          }
        }
      }
    }
    
    val m = new Model
    val paramNames = IndexedList.create[String]
    val startParam = m.add_parameters(Seq(vocab.length))
    paramNames.add("start")
    val transitionParam = m.add_parameters(Seq(vocab.length * vocab.length))
    paramNames.add("transition")
    val model = new PpModel(paramNames, Array(startParam, transitionParam),
        IndexedList.create[String], Array(), m, true)
    
    val examples = List(
        PpExample(lm(3), lm(3), Env.init, makeOracle(Array(0,1,0))),
        PpExample(lm(3), lm(3), Env.init, makeOracle(Array(0,1,2)))
    )

    val sgd = new SimpleSGDTrainer(model.model, 0.1f, 0.1f)
    // val trainer = new LoglikelihoodTrainer(1000, 100, model, sgd, new NullLogFunction())
    val trainer = new GlobalLoglikelihoodTrainer(1000, 100, model, sgd, new NullLogFunction())
    
    trainer.train(examples)

    /*
    val env = Env.init
    val computationGraph = new ComputationGraph
    val marginals = lm(1).beamSearch(100, env, model.getInitialComputationGraph(computationGraph))
    val values = marginals.executions
    val partitionFunction = marginals.partitionFunction
    values.length should be(2)
    values(0).value should be(List(1))
    (values(0).prob / partitionFunction) should be(0.8 +- TOLERANCE)
    
    computationGraph.delete()
    */
  }

  it should "learn xor" in {
    def xor(left: Boolean, right: Boolean): Pp[Boolean] = {
      // Build a feature vector from the inputs
      val values = Array.ofDim[Float](2)
      values(0) = if (left) { 1 } else { 0 }
      values(1) = if (right) { 1 } else { 0 }
      val featureVector = new FloatVector(2)
      featureVector.set(0, values(0))
      featureVector.set(1, values(1))

      for {
        // Build a 2 layer neural network with a tanh
        // nonlinearity.
        params <- Pp.param("params")
        bias <- Pp.param("bias")
        featureVectorExpression <- Pp.constant(Seq(2), featureVector)
        hidden = tanh((params * featureVectorExpression) + bias)
        params2 <- Pp.param("params2")
        bias2 <- Pp.param("bias2")
        dist = (params2 * hidden) + bias2

        // Choose the output nondeterministically according to 
        // the per-class scores generated by the neural network.
        y <- Pp.choose(Array(false, true), dist)

        // Extraneous values in the computation graph
        // shouldn't cause problems.
        // foo = dist * dist
      } yield {
        y
      }
    }
    
    // Initialize model parameters.
    val m = new Model
    val paramNames = IndexedList.create[String]
    val params = ListBuffer[Parameter]()
    paramNames.add("params")
    params += m.add_parameters(Seq(8, 2))
    paramNames.add("bias")
    params += m.add_parameters(Seq(8))
    paramNames.add("params2")
    params += m.add_parameters(Seq(2, 8))
    paramNames.add("bias2")
    params += m.add_parameters(Seq(2))
    
    val model = new PpModel(paramNames, params.toArray,
        IndexedList.create[String], Array(), m, true)

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
    val sgd = new SimpleSGDTrainer(model.model, 0.1f, 0.01f)
    val trainer = new LoglikelihoodTrainer(1000, 100, model, sgd, new NullLogFunction())
    trainer.train(examples)

    // Check that training error is zero.
    for (ex <- data) {
      val env = Env.init
      val cg = new ComputationGraph
      val marginals = xor(ex._1, ex._2).beamSearch(100, env, model.getInitialComputationGraph(cg))
      val values = marginals.executions
      val partitionFunction = marginals.partitionFunction

      values(0).value should be(ex._3)
      (values(0).prob / partitionFunction) should be(1.0 +- 0.1)
      cg.delete()
    }
  }
}
