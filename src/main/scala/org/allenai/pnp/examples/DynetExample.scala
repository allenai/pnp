package org.allenai.pnp.examples

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._

object DynetExample {
  val HIDDEN_SIZE = 8
  val ITERATIONS = 30

  import DynetScalaHelpers._

  def main(args: Array[String]) {
    println("Basic XOR example")
    myInitialize()
    val m = new Model
    val sgd = new SimpleSGDTrainer(m)
    val cg = new ComputationGraph

    val param_W = m.add_parameters(Seq(HIDDEN_SIZE, 2))
    val param_b = m.add_parameters(Seq(HIDDEN_SIZE))
    val param_V = m.add_parameters(Seq(1, HIDDEN_SIZE))
    val param_a = m.add_parameters(Seq(1))

    val node_W = parameter(cg, param_W)
    val node_b = parameter(cg, param_b)
    val node_V = parameter(cg, param_V)
    val node_a = parameter(cg, param_a)

    // Vector handling needs to be streamlined!
    val x_values = new FloatVector(2)
    x_values.set(1, 1.5f)
    x_values.set(1, 2.6f)
    println("x_values(0) = " + x_values.get(0))
    val node_x = input(cg, Seq(2), x_values)
    var y_value = 0f
    val node_y = input(cg, y_value)

    val node_h = tanh(node_W * node_x + node_b)
    val node_y_pred = node_V * node_h + node_a
    val node_loss = squared_distance(node_y_pred, node_y)

    val out_y_pred = cg.forward(node_y_pred)
    println(s"out_y_pred(0) = ${as_vector(out_y_pred).get(0)}")

    // Only compute the required node, this should not recompute y_pred
    val out_loss = cg.incremental_forward(node_loss)
    println(s"out_loss = ${as_scalar(out_loss)}")

    // Use affine_transformation
    val args = new ExpressionVector(3)
    args.set(0, node_b)
    args.set(1, node_W)
    args.set(2, node_x)
    val node_h2 = tanh(affine_transform_VE(args))
    val node_y2_pred = node_V * node_h2 + node_a
    val out_y2_pred = cg.forward(node_y2_pred)
    println(s"out_y2_pred(0) = ${as_vector(out_y2_pred).get(0)}")

    cg.delete()

  }
}