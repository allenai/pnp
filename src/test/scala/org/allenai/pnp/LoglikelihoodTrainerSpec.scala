package org.allenai.pnp

import scala.collection.JavaConverters._
import org.scalatest._
import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._
import org.allenai.pnp.examples.DynetScalaHelpers._
import com.jayantkrish.jklol.util.IndexedList
import com.jayantkrish.jklol.training.NullLogFunction
import scala.collection.mutable.ListBuffer

class LoglikelihoodTrainerSpec extends FlatSpec with Matchers {
  
  initialize(new DynetParams())

  val TOLERANCE = 0.01
  
  import LoglikelihoodTrainerSpec._

  "LoglikelihoodTrainer" should "train sequence models" in {
    val model = fooModel()
    val examples = List(
        PpExample.fromDistributions(foo(2, null), foo(2, List(1, 0))),
        PpExample.fromDistributions(foo(3, null), foo(3, List(1, 1, 1)))
    )

    val sgd = new SimpleSGDTrainer(model.model, 0.1f, 0.1f)
    val trainer = new LoglikelihoodTrainer(1000, 100, false, model, sgd, new NullLogFunction())
    trainer.train(examples)

    val env = Env.init
    val computationGraph = new ComputationGraph
    val marginals = foo(1, null).beamSearch(100, env, model.getInitialComputationGraph(computationGraph))
    val values = marginals.executions
    val partitionFunction = marginals.partitionFunction
    values.length should be(2)
    values(0).value should be(List(1))
    (values(0).prob / partitionFunction) should be(0.8 +- TOLERANCE)
    values(1).value should be(List(0))
    (values(1).prob / partitionFunction) should be(0.2 +- TOLERANCE)
    
    computationGraph.delete()
  }

  it should "learn xor" in {
    val model = xorModel()

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
    val trainer = new LoglikelihoodTrainer(1000, 100, false, model, sgd, new NullLogFunction())
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
  
  it should "learn with multiple correct labels" in {
    val model = xorModel()

    // Create training data.
    val data = List(
      (true, true, List(false, true)),
      (true, false, List(true)),
      (false, true, List(true)),
      (false, false, List(false))
    )
    val examples = data.map(x => {
      val unconditional = xor(x._1, x._2)
      val conditional = unconditional
      
      val score = new ExecutionScore() {
        def apply(tag: Any, choice: Any, env: Env): Double = {
          if ("choice".equals(tag)) {
            if (x._3.contains(choice.asInstanceOf[Boolean])) { 
              0.0
            } else {
              Double.NegativeInfinity
            }
          } else {
            0.0
          }
        }
      }
      
      PpExample(unconditional, conditional, Env.init, score)
    })

    // Train model
    val sgd = new SimpleSGDTrainer(model.model, 0.1f, 0.01f)
    val trainer = new LoglikelihoodTrainer(1000, 100, true, model, sgd, new NullLogFunction())
    trainer.train(examples)

    // Check that training error is zero.
    for (ex <- data) {
      val env = Env.init
      val cg = new ComputationGraph
      val marginals = xor(ex._1, ex._2).beamSearch(100, env, model.getInitialComputationGraph(cg))
      val values = marginals.executions
      val partitionFunction = marginals.partitionFunction

      val expectedProb = 1.0 / ex._3.length 
      for (expected <- ex._3) {
        val executions = values.filter(x => x.value == expected)
        executions.length should be(1)

        val execution = executions(0)
        (execution.prob / partitionFunction) should be(expectedProb +- 0.1)
      }
      cg.delete()
    }
  }
}

object LoglikelihoodTrainerSpec {
  def foo(k: Int, label: List[Int]): Pp[List[Int]] = {
    if (k == 0) {
      Pp.value(List.empty[Int])
    } else {
      for {
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
      } yield {
        x :: y
      }
    }
  }
  
  def fooModel(): PpModel = {
    val m = new Model
    val paramNames = IndexedList.create[String]
    val flipParam = m.add_parameters(Seq(2))
    paramNames.add("flip")
    new PpModel(paramNames, Array(flipParam),
        IndexedList.create[String], Array(), m, true)
  }
  
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
      y <- Pp.choose(Array(false, true), dist, "choice")

      // Extraneous values in the computation graph
      // shouldn't cause problems.
      // foo = dist * dist
    } yield {
      y
    }
  }
  
  def xorModel(): PpModel = {
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

    new PpModel(paramNames, params.toArray,
        IndexedList.create[String], Array(), m, true)
  }
}
