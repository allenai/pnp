package org.allenai.pnp

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

import com.google.common.base.Preconditions
import com.google.common.base.Supplier
import com.jayantkrish.jklol.tensor.DenseTensor
import com.jayantkrish.jklol.tensor.SparseTensor
import com.jayantkrish.jklol.tensor.Tensor
import com.jayantkrish.jklol.training.LogFunction
import com.jayantkrish.jklol.training.NullLogFunction
import com.jayantkrish.jklol.util.KbestQueue
import com.jayantkrish.jklol.util.ObjectPool

/** Neural probabilistic program monad. Pp[X] represents a
  * function from neural network parameters to a probabilistic
  * computation that nondeterministically executes to values
  * of type X. Each execution has an associated score proportional
  * to its probability and its own environment containing mutable
  * global state.
  *
  * A program can also retrieve neural network parameters
  * and construct a computation graph whose forward-pass values
  * may influence the probabilities of its executions. Similarly,
  * the nondeterministic choices made within the program may
  * influence the structure of the computation graph.
  *
  * Neural probabilistic programs are constructed and manipulated
  * using for/yield comprehensions and the functions in the
  * Pp object.
  */
sealed trait Pp[A] {
  // Methods that must be overriden in implementing classes.

  /** flatMap is the monad's bind operator. It chains two
    * probabilistic computations together in the natural way
    * where f represents a conditional distribution P(B | A).
    * Hence, binding f to a distribution P(A) returns the
    * marginal distribution over B, sum_a P(A=a) P(B | A).
    */
  def flatMap[B](f: A => Pp[B]): Pp[B]

  /** Implements a single search step of beam search.
    */
  def searchStep(env: Env, logProb: Double, continuation: (A, Env, Double) => Unit,
    queue: PpSearchQueue): Unit

  // Methods that do not need to be overriden

  def map[B](f: A => B): Pp[B] = flatMap { a => Pp.value(f(a)) }

  /** Performs a beam search over executions of this program returning
    * at most beamSize execution results. env is the initial global
    * state of the program, and graph is the initial computation graph.
    * These must contain values for any global variables or neural network
    * parameters referenced in the program.
    */
  def beamSearch(beamSize: Int, env: Env, stateCostArg: Env => Double,
    graph: CompGraph, log: LogFunction): PpBeamMarginals[A] = {

    val stateCost = if (stateCostArg == null) {
      (e: Env) => 0.0
    } else {
      stateCostArg
    }

    val queue = new BeamPpSearchQueue(beamSize, stateCost, graph, log)
    val finished = new KbestQueue(beamSize, Array.empty[(A, Env)])
    def finalContinuation(value: A, e: Env, logProb: Double): Unit = {
      val stateLogProb = stateCost(e) + logProb
      if (stateLogProb > Double.NegativeInfinity) {
        finished.offer((value, e), stateLogProb)
      }
    }

    val startEnv = env.setLog(log)
    this.searchStep(startEnv, 0.0, finalContinuation, queue)

    val beam = new Array[Searchable](beamSize)
    val beamScores = new Array[Double](beamSize)
    var numIters = 0
    while (queue.queue.size > 0) {
      numIters += 1
      // println(numIters + " " + queue.queue.size)

      val beamSize = queue.queue.size
      Array.copy(queue.queue.getItems, 0, beam, 0, beamSize)
      Array.copy(queue.queue.getScores, 0, beamScores, 0, beamSize)
      queue.queue.clear

      for (i <- 0 until beamSize) {
        beam(i).searchStep()
      }
    }

    // println(numIters)

    val numFinished = finished.size
    val finishedItems = finished.getItems.slice(0, numFinished)
    val finishedScores = finished.getScores.slice(0, numFinished)

    val executions = finishedItems.zip(finishedScores).sortBy(x => -1 * x._2).map(
      x => new Execution(x._1._1, x._1._2, x._2)
    )

    new PpBeamMarginals(executions, queue.graph, numIters)
  }

  def beamSearchWithFilter(beamSize: Int, env: Env, keepState: Env => Boolean,
    graph: CompGraph, log: LogFunction): PpBeamMarginals[A] = {
    def cost(e: Env): Double = {
      if (keepState(e)) {
        0.0
      } else {
        Double.NegativeInfinity
      }
    }
    beamSearch(beamSize, env, cost _, graph, log)
  }

  // Version of beam search for programs that don't have trainable
  // parameters
  def beamSearch(k: Int): Seq[(A, Double)] = {
    beamSearch(k, Env.init).executions.map(x => (x.value, x.prob))
  }

  def beamSearch(k: Int, env: Env): PpBeamMarginals[A] = {
    beamSearchWithFilter(k, env, (x: Env) => true)
  }

  def beamSearch(k: Int, env: Env, cg: CompGraph): PpBeamMarginals[A] = {
    beamSearch(k, env, (x: Env) => 0.0, cg, new NullLogFunction())
  }

  def beamSearchWithFilter(
    k: Int, env: Env, keepState: Env => Boolean, cg: CompGraph
  ): PpBeamMarginals[A] = {
    beamSearchWithFilter(k, env, keepState, cg, new NullLogFunction())
  }

  def beamSearchWithFilter(k: Int, env: Env, keepState: Env => Boolean): PpBeamMarginals[A] = {
    beamSearchWithFilter(k, env, keepState, null, new NullLogFunction())
  }

  def inOneStep(): Pp[A] = {
    CollapsedSearch(this)
  }
}

case class ValuePp[A](value: A) extends Pp[A] {
  override def flatMap[B](f: A => Pp[B]) = f(value)

  override def searchStep(env: Env, logProb: Double, continuation: (A, Env, Double) => Unit,
    queue: PpSearchQueue) = {
    continuation(value, env, logProb)
  }
}

case class ScorePp(score: Double) extends Pp[Unit] {
  override def flatMap[B](f: Unit => Pp[B]) = BindPp(this, f)

  override def searchStep(env: Env, logProb: Double, continuation: (Unit, Env, Double) => Unit,
    queue: PpSearchQueue) = {
    continuation((), env, logProb + Math.log(score))
  }
}

/** Categorical distribution representing a nondeterministic
  * choice of an element of dist. The elements of dist are
  * scores, i.e., log probabilities.
  */
case class CategoricalPp[A](dist: Seq[(A, Double)]) extends Pp[A] {
  override def flatMap[B](f: A => Pp[B]) = BindPp(this, f)

  override def searchStep(env: Env, logProb: Double, continuation: (A, Env, Double) => Unit,
    queue: PpSearchQueue) = {
    dist.foreach(x => queue.offer(x._1, env, continuation, logProb + x._2, env))
  }
}

case class BindPp[A, C](b: Pp[C], f: C => Pp[A]) extends Pp[A] {
  override def flatMap[B](f: A => Pp[B]) = BindPp(this, f)

  override def searchStep(env: Env, logProb: Double, continuation: (A, Env, Double) => Unit,
    queue: PpSearchQueue) = {
    b.searchStep(env, logProb, (x, e, d) => f(x).searchStep(e, d, continuation, queue), queue)
  }
}

case class GetEnv() extends Pp[Env] {
  override def flatMap[B](f: Env => Pp[B]) = BindPp(this, f)

  override def searchStep(env: Env, logProb: Double, continuation: (Env, Env, Double) => Unit,
    queue: PpSearchQueue) = {
    continuation(env, env, logProb)
  }
}

case class SetEnv(nextEnv: Env) extends Pp[Unit] {
  override def flatMap[B](f: Unit => Pp[B]) = BindPp(this, f)

  override def searchStep(env: Env, logProb: Double, continuation: (Unit, Env, Double) => Unit,
    queue: PpSearchQueue) = {
    continuation((), nextEnv, logProb)
  }
}

case class SetVar(name: String, value: AnyRef) extends Pp[Unit] {
  override def flatMap[B](f: Unit => Pp[B]) = BindPp(this, f)

  override def searchStep(env: Env, logProb: Double, continuation: (Unit, Env, Double) => Unit,
    queue: PpSearchQueue) = {
    val nextEnv = env.setVar(name, value)
    continuation((), nextEnv, logProb)
  }
}

case class SetVarInt(nameInt: Int, value: AnyRef) extends Pp[Unit] {
  override def flatMap[B](f: Unit => Pp[B]) = BindPp(this, f)

  override def searchStep(env: Env, logProb: Double, continuation: (Unit, Env, Double) => Unit,
    queue: PpSearchQueue) = {
    val nextEnv = env.setVar(nameInt, value)
    continuation((), nextEnv, logProb)
  }
}

// Class for collapsing out multiple choices into a single choice
case class CollapsedSearch[A](dist: Pp[A]) extends Pp[A] {
  override def flatMap[B](f: A => Pp[B]) = BindPp(this, f)

  override def searchStep(env: Env, logProb: Double, continuation: (A, Env, Double) => Unit,
    queue: PpSearchQueue) = {
    def finalContinuation(value: A, e: Env, logProb: Double): Unit = {
      queue.offer(value, e, continuation, logProb, env)
    }

    val nextQueue = new EnumeratePpSearchQueue(queue.stateCost, queue.graph, queue.log)
    dist.searchStep(env, logProb, finalContinuation, nextQueue)
  }
}

// Classes for representing computation graph elements.

case class ParameterPp(name: String, id: Int) extends Pp[CompGraphNode] {
  override def flatMap[B](f: CompGraphNode => Pp[B]) = BindPp(this, f)

  override def searchStep(env: Env, logProb: Double,
    continuation: (CompGraphNode, Env, Double) => Unit, queue: PpSearchQueue) = {
    val node = if (name == null) {
      queue.graph.get(id)
    } else {
      queue.graph.getParameter(name)
    }
    continuation(node, env, logProb)
  }
}

case class ConstantTensorPp(tensor: Tensor) extends Pp[CompGraphNode] {
  override def flatMap[B](f: CompGraphNode => Pp[B]) = BindPp(this, f)

  override def searchStep(env: Env, logProb: Double,
    continuation: (CompGraphNode, Env, Double) => Unit, queue: PpSearchQueue) = {
    val graph = queue.graph
    val constNode = new ConstantNode(graph.nextId(), graph, tensor)
    graph.addNode(constNode)
    continuation(constNode, env, logProb)
  }
}

case class ParameterizedCategoricalPp[A](items: Array[A], parameter: CompGraphNode,
    keyPrefix: Array[Int]) extends Pp[A] {
  override def flatMap[B](f: A => Pp[B]) = BindPp(this, f)

  override def searchStep(env: Env, logProb: Double, continuation: (A, Env, Double) => Unit,
    queue: PpSearchQueue) = {
    val paramTensor = parameter.value
    val startKeyNum = paramTensor.dimKeyPrefixToKeyNum(keyPrefix)
    val endKeyNum = if (keyPrefix.length == 0) {
      paramTensor.getMaxKeyNum
    } else {
      startKeyNum + (paramTensor.getDimensionOffsets())(keyPrefix.length - 1)
    }
    val numTensorValues = (endKeyNum - startKeyNum).asInstanceOf[Int]

    Preconditions.checkState(
      numTensorValues == items.length,
      "parameter dimensionality %s doesn't match item's %s (%s)",
      numTensorValues.asInstanceOf[AnyRef], items.length.asInstanceOf[AnyRef], items
    )

    for (i <- 0 until numTensorValues) {
      val keyNum = startKeyNum + i
      val nextEnv = env.addLabel(parameter, makeLabelIndicator(keyNum, paramTensor))
      queue.offer(items(i), nextEnv, continuation, logProb + paramTensor.get(keyNum), env)
    }
  }

  def makeLabelIndicator(keyNum: Long, params: Tensor): Tensor = {
    SparseTensor.singleElement(
      params.getDimensionNumbers,
      params.getDimensionSizes, params.keyNumToDimKey(keyNum), 1.0
    )
  }
}

case class ParameterizedArrayCategoricalPp[A](items: Array[A], parameters: Array[CompGraphNode]) extends Pp[A] {
  Preconditions.checkArgument(
    items.length == parameters.length,
    "Expected arrays to be equal length: items (%s) and parameters (%s)",
    items.length.asInstanceOf[AnyRef], parameters.length.asInstanceOf[AnyRef]
  )

  override def flatMap[B](f: A => Pp[B]) = BindPp(this, f)

  override def searchStep(env: Env, logProb: Double, continuation: (A, Env, Double) => Unit,
    queue: PpSearchQueue) = {
    val paramValues = parameters.map(x => {
      Preconditions.checkState(x.value.getDimensionNumbers.length == 0)
      x.value.get(0)
    })

    for (i <- 0 until items.length) {
      val nextEnv = env.addLabel(parameters(i), DenseTensor.scalar(1.0))
      queue.offer(items(i), nextEnv, continuation, logProb + paramValues(i), env)
    }
  }
}

case class StartTimerPp(timerName: String) extends Pp[Unit] {
  override def flatMap[B](f: Unit => Pp[B]) = BindPp(this, f)

  override def searchStep(env: Env, logProb: Double, continuation: (Unit, Env, Double) => Unit,
    queue: PpSearchQueue) = {
    queue.offer((), env.startTimer(timerName), continuation, logProb, env)
  }
}

case class StopTimerPp(timerName: String) extends Pp[Unit] {
  override def flatMap[B](f: Unit => Pp[B]) = BindPp(this, f)

  override def searchStep(env: Env, logProb: Double, continuation: (Unit, Env, Double) => Unit,
    queue: PpSearchQueue) = {
    queue.offer((), env.stopTimer(timerName), continuation, logProb, env)
  }
}

// Search queue items for beam search   
sealed trait Searchable {
  def searchStep(): Unit
}

case class SearchState[A](value: A, env: Env, logProb: Double, continuation: (A, Env, Double) => Unit) extends Searchable {
  def searchStep(): Unit = {
    env.resumeTimers()
    continuation.apply(value, env, logProb)
    env.pauseTimers()
  }
}

class SearchState2(var value: AnyRef, var env: Env, var logProb: Double,
    var continuation: (AnyRef, Env, Double) => Unit) extends Searchable {

  def searchStep(): Unit = {
    env.resumeTimers()
    continuation.apply(value, env, logProb)
    env.pauseTimers()
  }
}

sealed trait PpSearchQueue {
  val graph: CompGraph
  val stateCost: Env => Double
  val log: LogFunction

  def offer[A](value: A, env: Env, continuation: (A, Env, Double) => Unit,
    logProb: Double, myEnv: Env): Unit
}

class BeamPpSearchQueue(size: Int, val stateCost: Env => Double,
    val graph: CompGraph, val log: LogFunction) extends PpSearchQueue {

  val supplier = new Supplier[SearchState2]() {
    def get: SearchState2 = {
      new SearchState2(null, null, 0.0, null)
    }
  }

  val pool = new ObjectPool(supplier, size + 1, Array.empty[SearchState2])
  val queue = new KbestQueue(size, Array.empty[Searchable])

  override def offer[A](value: A, env: Env, continuation: (A, Env, Double) => Unit,
    logProb: Double, myEnv: Env): Unit = {
    val stateLogProb = stateCost(env) + logProb
    if (stateLogProb > Double.NegativeInfinity) {
      /*
      val next = pool.alloc()
      next.value = value
      next.env = env
      next.continuation = continuation
      next.logProb = logProb
      val dequeued = queue.offer(next, logProb)

      if (dequeued != null) {
        pool.dealloc(dequeued)
      }
      */
      queue.offer(SearchState(value, env, stateLogProb, continuation), logProb)
    }
  }
}

class EnumeratePpSearchQueue(
    val stateCost: Env => Double,
    val graph: CompGraph, val log: LogFunction
) extends PpSearchQueue {
  override def offer[A](value: A, env: Env, continuation: (A, Env, Double) => Unit,
    logProb: Double, myEnv: Env): Unit = {
    myEnv.pauseTimers()
    val stateLogProb = stateCost(env) + logProb
    if (stateLogProb > Double.NegativeInfinity) {
      env.resumeTimers()
      continuation(value, env, stateLogProb)
      env.pauseTimers()
    }
    myEnv.resumeTimers()
  }
}

class Execution[A](val value: A, val env: Env, val logProb: Double) {
  def prob = Math.exp(logProb)

  override def toString: String = {
    "[Execution " + value + " " + logProb + "]"
  }
}

class PpBeamMarginals[A](val executions: Seq[Execution[A]], val graph: CompGraph,
    val searchSteps: Int) {

  /*
  def logPartitionFunction(): Double = {
    executions.map(x => x.logProb).sum
  }
  */

  def partitionFunction(): Double = {
    executions.map(x => x.prob).sum
  }

  def condition(pred: (A, Env) => Boolean): PpBeamMarginals[A] = {
    return new PpBeamMarginals(executions.filter(x => pred(x.value, x.env)), graph, searchSteps)
  }
}

object Pp {

  /** Create a program that returns {@code value}
    */
  def value[A](value: A): Pp[A] = { ValuePp(value) }

  /** A nondeterministic choice. Creates a program
    * that chooses and returns a single value from
    * {@code dist} with the given probability.
    */
  def chooseMap[A](dist: Seq[(A, Double)]): Pp[A] = {
    CategoricalPp(dist.map(x => (x._1, Math.log(x._2))))
  }

  def choose[A](items: Seq[A], weights: Seq[Double]): Pp[A] = {
    CategoricalPp(items.zip(weights).map(x => (x._1, Math.log(x._2))))
  }
  
  def choose[A](items: Seq[A]): Pp[A] = {
    CategoricalPp(items.map(x => (x, 0.0)))
  }

  /** The failure program that has no executions.
    */
  def fail[A]: Pp[A] = { CategoricalPp(Seq.empty[(A, Double)]) }

  def require(value: Boolean): Pp[Unit] = {
    if (value) {
      Pp.value(())
    } else {
      Pp.fail
    }
  }

  // Methods for manipulating global program state

  /** Gets the environment (mutable state) of the program.
    * See the {@code getVar} method to get the value of a
    * single variable.
    */
  private def getEnv(): Pp[Env] = { GetEnv() }

  /** Sets the environment (mutable state) of the program.
    * See the {@code setVar} method to set the value of a
    * single variable.
    */
  private def setEnv(env: Env): Pp[Unit] = { SetEnv(env) }

  /** Gets the value of the variable {@code name} in the
    * program's environment.
    */
  def getVar[A](name: String): Pp[A] = {
    for {
      x <- getEnv
    } yield {
      x.getVar[A](name)
    }
  }

  def getVar[A](nameInt: Int): Pp[A] = {
    for {
      x <- getEnv
    } yield {
      x.getVar[A](nameInt)
    }
  }

  /** Sets the value of the variable {@code name} to
    * {@code value} in the program's environment.
    */
  def setVar[A <: AnyRef](name: String, value: A): Pp[Unit] = {
    SetVar(name, value)
  }

  def setVar[A <: AnyRef](nameInt: Int, value: A): Pp[Unit] = {
    SetVarInt(nameInt, value)
  }

  def isVarBound(name: String): Pp[Boolean] = {
    for {
      x <- getEnv
    } yield {
      x.isVarBound(name)
    }
  }

  // Methods for interacting with the computation graph

  /** Get a neural network parameter by name.
    */
  def param(name: String): Pp[CompGraphNode] = { ParameterPp(name, -1) }

  def param(id: Int): Pp[CompGraphNode] = { ParameterPp(null, id) }

  /** Add a tensor to the computation graph as a constant.
    */
  def constant(tensor: Tensor): Pp[CompGraphNode] = { ConstantTensorPp(tensor) }

  /** Chooses an item. The ith item's score is the
    * ith index in parameter.
    */
  def choose[A](items: Array[A], parameter: CompGraphNode): Pp[A] = {
    ParameterizedCategoricalPp(items, parameter, Array.emptyIntArray)
  }

  /** Chooses an item. The ith item's score is the ith
    * element of parameters, each of which is a scalar.
    */
  def choose[A](items: Array[A], parameters: Array[CompGraphNode]): Pp[A] = {
    ParameterizedArrayCategoricalPp(items, parameters)
  }

  def chooseSlice[A](items: Array[A], parameter: CompGraphNode,
    keyPrefix: Array[Int]): Pp[A] = {
    ParameterizedCategoricalPp(items, parameter, keyPrefix)
  }

  /** Add a scalar value to the score of this execution.
    */
  def score(parameter: CompGraphNode): Pp[Unit] = {
    ParameterizedCategoricalPp(Array(()), parameter, Array.emptyIntArray)
  }

  // Methods for timing execution
  def startTimer(name: String): Pp[Unit] = {
    StartTimerPp(name)
  }

  def stopTimer(name: String): Pp[Unit] = {
    StopTimerPp(name)
  }
}