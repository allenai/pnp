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
import org.allenai.pnp.PnpExample

import com.jayantkrish.jklol.training.NullLogFunction
import org.allenai.pnp.BsoTrainer

class MultilayerPerceptron {
  
  import MultilayerPerceptron._
  
  }

object MultilayerPerceptron {
  
  val FEATURE_VECTOR_DIM = 3
  val HIDDEN_DIM = 50
  val LABEL_DIM = 10
  
  def mlp(x: FloatVector): Pnp[Boolean] = {
    for {
      cg <- computationGraph()
      weights1 <- param("layer1Weights")
      bias1 <- param("layer1Bias")
      weights2 <- param("layer2Weights")

      inputExpression = Expression.input(Dim(FEATURE_VECTOR_DIM), x)
      scores = weights2 * Expression.tanh((weights1 * inputExpression) + bias1)

      y <- choose(Array(true, false), scores)
    } yield {
      y
    }
  }
  
  def labelNn(left: Boolean, right: Boolean, cg: CompGraph): Expression = {
    val leftParam = cg.getLookupParameter("left")
    val rightParam = cg.getLookupParameter("right")
    val leftVec = Expression.lookup(leftParam, if (left) { 0 } else { 1 })
    val rightVec = Expression.lookup(rightParam, if (right) { 0 } else { 1 })
    
    Expression.dotProduct(leftVec, rightVec)
  }
  
  def sequenceTag(xs: Seq[FloatVector]): Pnp[List[Boolean]] = {
    xs.foldLeft(Pnp.value(List[Boolean]()))((x, y) => for {
      cur <- mlp(y)
      rest <- x
  
      cg <- computationGraph()
      _ <- if (rest.length > 0) {
        score(labelNn(cur, rest.head, cg))
      } else {
        value(())
      }
    } yield {
      cur :: rest
    })
  }

  def main(args: Array[String]): Unit = {
    // Initialize dynet
    Initialize.initialize()

    val model = PnpModel.init(true)
    model.addParameter("layer1Weights", Dim(HIDDEN_DIM, FEATURE_VECTOR_DIM))
    model.addParameter("layer1Bias", Dim(HIDDEN_DIM))
    model.addParameter("layer2Weights", Dim(2, HIDDEN_DIM))
    
    val featureVector = new FloatVector(Seq(1.0f, 2f, 3f))
    val dist = mlp(featureVector)
    val marginals = dist.beamSearch(2, model)
 
    for (x <- marginals.executions) {
      println(x)
    }
    
    val featureVectors = Seq(featureVector, featureVector, featureVector)
    
    model.locallyNormalized = false
    model.addLookupParameter("left", 2, Dim(LABEL_DIM))
    model.addLookupParameter("right", 2, Dim(LABEL_DIM))
    val dist2 = sequenceTag(featureVectors)
    val marginals2 = dist2.beamSearch(5, model)
    for (x <- marginals2.executions) {
      println(x)
    }
    
    val flip: Pnp[Boolean] = choose(Array(true, false), Array(0.5, 0.5))
    val twoFlips: Pnp[Boolean] = for {
      x <- flip
      y <- flip
    } yield {
      x && y
    }
    val marginals3 = twoFlips.beamSearch(5)
    println(marginals3.marginals().getProbabilityMap)
  }
}