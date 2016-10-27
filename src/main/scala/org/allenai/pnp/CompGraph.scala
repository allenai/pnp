package org.allenai.pnp

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.PriorityQueue

import com.google.common.base.Preconditions
import com.jayantkrish.jklol.tensor.AbstractTensor
import com.jayantkrish.jklol.tensor.DenseTensorBuilder
import com.jayantkrish.jklol.tensor.SparseTensor
import com.jayantkrish.jklol.tensor.Tensor
import com.jayantkrish.jklol.util.IndexedList
import com.jayantkrish.jklol.util.IntMultimap

/** Computation graph of a neural network.
  */
class CompGraph(val paramNames: IndexedList[String], params: Array[Tensor]) {

  val nodes = ArrayBuffer[CompGraphNode]()
  val edges = IntMultimap.createWithInitialCapacity(100)
  val numOutboundEdges = ArrayBuffer[Int]()

  // Initialize the nodes of the graph with a node per
  // parameter.
  for (i <- 0 until params.length) {
    val id = nextId()
    addNode(new ParameterNode(id, this, paramNames.get(i), params(i)))
  }

  def nextId(): Int = {
    nodes.size
  }

  def addNode(node: CompGraphNode): Unit = {
    Preconditions.checkArgument(node.id == nodes.size)
    nodes += node
    numOutboundEdges += 0

    for (inbound <- node.dependsOn) {
      edges.put(inbound, node.id)
      numOutboundEdges(inbound) += 1
    }
  }

  def getOutboundEdges(id: Int): Array[Int] = {
    edges.getArray(id)
  }

  def getInboundEdges(id: Int): Array[Int] = {
    nodes(id).dependsOn
  }

  def getParameter(name: String): CompGraphNode = {
    nodes(paramNames.getIndex(name))
  }

  def get(id: Int): CompGraphNode = {
    nodes(id)
  }

  def backprop(): Backprop = {
    new Backprop(this)
  }
}

/** Data structure for performing backpropagation on a
  * computation graph. This class is initialized by
  * calling incrementGradient to add inbound gradients to
  * the computation graph, then calling backprop to
  * propagate these gradients through the entire graph.
  */
class Backprop(val graph: CompGraph) {

  val gradients = Array.fill[DenseTensorBuilder](graph.nodes.size) { null }
  val queue = new PriorityQueue[Int]()
  val offered = Array.fill[Boolean](gradients.size) { false }

  def getGradientBuilder(id: Int): DenseTensorBuilder = {
    if (gradients(id) == null) {
      val tensor = graph.nodes(id).value
      gradients(id) = new DenseTensorBuilder(tensor.getDimensionNumbers, tensor.getDimensionSizes);
      queue += id
    }
    gradients(id)
  }

  def incrementGradient(id: Int, tensor: Tensor): Unit = {
    incrementGradient(id, tensor, 1.0)
  }

  def incrementGradient(id: Int, tensor: Tensor, multiplier: Double): Unit = {
    val builder = getGradientBuilder(id)
    builder.incrementWithMultiplier(tensor, multiplier)
  }

  def backprop(): Unit = {
    var numVerticesProcessed = 0
    while (queue.size > 0) {
      val id = queue.dequeue
      val node = graph.get(id)
      numVerticesProcessed += 1

      val gradientBuilder = gradients(id)
      if (gradientBuilder != null) {
        val gradient = gradients(id).buildNoCopy()
        node.backprop(gradient, this)
      }
    }

    // println(numVerticesProcessed)
  }

  def getGradient(id: Int): Tensor = {
    if (gradients(id) != null) {
      gradients(id).buildNoCopy()
    } else {
      null
    }
  }
}

/** A node in a computation graph. Nodes have tensor values and
  * override math operators for performing tensor math. These
  * operators generate new nodes in the computation graph.
  */
sealed trait CompGraphNode {
  val id: Int
  val value: Tensor
  val graph: CompGraph
  val dependsOn: Array[Int]
  val isConstant: Boolean

  def backprop(inbound: Tensor, backprop: Backprop): Unit

  def *(smaller: Tensor): CompGraphNode = {
    val constNode = new ConstantNode(graph.nextId(), graph, smaller)
    graph.addNode(constNode)
    this * constNode
  }

  def *(smaller: CompGraphNode): CompGraphNode = {
    val node = new InnerProductNode(graph.nextId(), graph, this, smaller)
    graph.addNode(node)
    node
  }

  def elementwiseProduct(smaller: CompGraphNode): CompGraphNode = {
    val node = new ProductNode(graph.nextId(), graph, this, smaller)
    graph.addNode(node)
    node
  }

  def outerProduct(right: CompGraphNode): CompGraphNode = {
    val node = new OuterProductNode(graph.nextId(), graph, this, right)
    graph.addNode(node)
    node
  }

  def +(other: CompGraphNode): CompGraphNode = {
    val node = new PlusNode(graph.nextId(), graph, this, other)
    graph.addNode(node)
    node
  }

  def tanh: CompGraphNode = {
    val node = new TanhNode(graph.nextId(), graph, this)
    graph.addNode(node)
    node
  }
}

abstract class AbstractCompGraphNode(val id: Int, val graph: CompGraph,
    val dependsOn: Array[Int], val isConstant: Boolean) extends CompGraphNode {
  Preconditions.checkArgument(dependsOn.find(_ >= id).isEmpty)
}

/** Inner product of two computation graph nodes. This operation
  * generalizes vector-vector and matrix-vector multiplication to
  * tensors.
  */
class InnerProductNode(id: Int, graph: CompGraph,
  val big: CompGraphNode, val small: CompGraphNode)
    extends AbstractCompGraphNode(id, graph, Array(big.id, small.id),
      big.isConstant && small.isConstant) {

  // True if this node represents the analog of a matrix-vector
  // product.
  val mvProduct = big.value.getDimensionNumbers.containsSlice(
    small.value.getDimensionNumbers
  )

  override val value = if (mvProduct) {
    big.value.innerProduct(small.value)
  } else {
    big.value.matrixInnerProduct(small.value)
  }

  override def backprop(inbound: Tensor, backprop: Backprop): Unit = {
    if (mvProduct) {
      if (!small.isConstant) {
        val builder = backprop.getGradientBuilder(small.id)
        builder.incrementInnerProductWithMultiplier(big.value, inbound, 1.0)
      }

      if (!big.isConstant) {
        val builder = backprop.getGradientBuilder(big.id)
        builder.incrementOuterProductWithMultiplier(inbound, small.value, 1.0)
      }
    } else {
      if (!small.isConstant) {
        val builder = backprop.getGradientBuilder(small.id)
        AbstractTensor.innerProduct(inbound, big.value, null, builder)
      }

      if (!big.isConstant) {
        val builder = backprop.getGradientBuilder(big.id)
        AbstractTensor.innerProduct(inbound, small.value, null, builder)
      }
    }
  }
}

class OuterProductNode(id: Int, graph: CompGraph, val left: CompGraphNode,
  val right: CompGraphNode) extends AbstractCompGraphNode(id, graph,
  Array(left.id, right.id), left.isConstant && right.isConstant) {

  override val value = left.value.outerProduct(right.value)

  override def backprop(inbound: Tensor, backprop: Backprop): Unit = {
    if (!left.isConstant) {
      val builder = backprop.getGradientBuilder(left.id)
      builder.incrementInnerProductWithMultiplier(inbound, right.value, 1.0)
    }

    if (!right.isConstant) {
      val builder = backprop.getGradientBuilder(right.id)
      builder.incrementInnerProductWithMultiplier(inbound, left.value, 1.0)
    }
  }
}

/** Elementwise product of two nodes. The values of both
  * nodes must have the same dimensions.
  */
class ProductNode(id: Int, graph: CompGraph,
  val big: CompGraphNode, val small: CompGraphNode)
    extends AbstractCompGraphNode(id, graph, Array(big.id, small.id),
      big.isConstant && small.isConstant) {

  override val value = big.value.elementwiseProduct(small.value)

  override def backprop(inbound: Tensor, backprop: Backprop): Unit = {
    if (!big.isConstant) {
      val builder = backprop.getGradientBuilder(big.id)
      builder.increment(inbound.elementwiseProduct(small.value))
    }

    if (!small.isConstant) {
      val builder = backprop.getGradientBuilder(small.id)
      builder.increment(inbound.elementwiseProduct(big.value))
    }
  }
}

/** Elementwise addition of two computation graph nodes.
  */
class PlusNode(id: Int, graph: CompGraph,
  val left: CompGraphNode, val right: CompGraphNode)
    extends AbstractCompGraphNode(id, graph, Array(left.id, right.id),
      left.isConstant && right.isConstant) {

  override val value = left.value.elementwiseAddition(right.value)

  override def backprop(inbound: Tensor, backprop: Backprop): Unit = {
    backprop.incrementGradient(left.id, inbound)
    backprop.incrementGradient(right.id, inbound)
  }
}

/** Elementwise hyperbolic tangent.
  */
class TanhNode(id: Int, graph: CompGraph, val node: CompGraphNode)
    extends AbstractCompGraphNode(id, graph, Array(node.id),
      node.isConstant) {

  override val value = node.value.elementwiseTanh()

  override def backprop(inbound: Tensor, backprop: Backprop): Unit = {
    val builder = backprop.getGradientBuilder(node.id)
    val maxKeyNum = builder.getMaxKeyNum
    for (keyNum <- 0L until maxKeyNum) {
      val v = value.get(keyNum)
      val i = inbound.get(keyNum)

      // The gradient of tanh(v) is 1 - v^2.
      builder.incrementEntryByKeyNum(i * (1 - v * v), keyNum)
    }
  }
}

/*
class MaxNode(id: Int, graph: CompGraph, val nodes: Array[CompGraphNode])
    extends AbstractCompGraphNode(id, graph, nodes.map(_.id).toArray,
      nodes.map(_.isConstant).fold(true)(_ && _)) {
  
}
*/

class StackNode(id: Int, graph: CompGraph, val nodes: Array[CompGraphNode], val dim: Int)
    extends AbstractCompGraphNode(id, graph, nodes.map(_.id).toArray,
      nodes.map(_.isConstant).fold(true)(_ && _)) {

  val tensors = nodes.map(_.value)
  val myDims = new ArrayBuffer[Int]
  val mySizes = new ArrayBuffer[Int]
  myDims += dim
  mySizes += nodes.length
  myDims ++= tensors(0).getDimensionNumbers
  mySizes ++= tensors(0).getDimensionSizes
  val builder = new DenseTensorBuilder(myDims.toArray, mySizes.toArray)

  val nodeDims = Array(dim)
  val nodeDimsSizes = Array(nodes.length)
  for (i <- 0 until nodes.length) {
    val node = nodes(i)
    val dimIndicator = SparseTensor.singleElement(nodeDims, nodeDimsSizes, Array(i), 1.0)
    builder.incrementOuterProductWithMultiplier(dimIndicator, node.value, 1.0)
  }
  override val value = builder.build()

  override def backprop(inbound: Tensor, backprop: Backprop): Unit = {
    for (i <- 0 until nodes.length) {
      val node = nodes(i)
      if (!node.isConstant) {
        val builder = backprop.getGradientBuilder(node.id)
        val nodeGradient = inbound.slice(nodeDims, Array(i))
        builder.increment(nodeGradient)
      }
    }
  }
}

class ParameterNode(id: Int, graph: CompGraph, val name: String, override val value: Tensor)
    extends AbstractCompGraphNode(id, graph, Array.empty[Int], false) {

  override def backprop(inbound: Tensor, backprop: Backprop): Unit = {
    // Don't need to do anything.
  }
}

class ConstantNode(id: Int, graph: CompGraph, override val value: Tensor)
    extends AbstractCompGraphNode(id, graph, Array.empty[Int], true) {

  override def backprop(inbound: Tensor, backprop: Backprop): Unit = {
    // Don't need to do anything.
  }
}
