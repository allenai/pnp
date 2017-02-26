package org.allenai.pnp.examples

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

import org.allenai.pnp.CompGraph
import org.allenai.pnp.Env
import org.allenai.pnp.ExecutionScore
import org.allenai.pnp.Pnp
import org.allenai.pnp.Pnp._
import org.allenai.pnp.PnpModel

import com.google.common.base.Preconditions
import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.dynet._
import edu.cmu.dynet.DynetScalaHelpers._
import edu.cmu.dynet.dynet_swig._
import org.allenai.pnp.PnpExample
import com.jayantkrish.jklol.training.NullLogFunction
import org.allenai.pnp.BsoTrainer

class MultilayerPerceptron {
  
  val FEATURE_VECTOR_DIM = 50
  
  def mlp(x: FloatVector): Pnp[Boolean] = {
    for {
      cg <- computationGraph()
      weights1 <- param("layer1Weights")
      bias1 <- param("layer1Bias")
      weights2 <- param("layer1Weights")

      inputExpression = input(cg.cg, Seq(FEATURE_VECTOR_DIM), x)
      scores = weights2 * tanh((weights1 * inputExpression) + bias1)

      y <- choose(Array(true, false), scores)
    } yield {
      y
    }
  }

  
}