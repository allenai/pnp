package org.allenai.pnp

import java.util.Arrays

import com.google.common.base.Preconditions
import com.jayantkrish.jklol.tensor.DenseTensor
import com.jayantkrish.jklol.tensor.SparseTensor
import com.jayantkrish.jklol.tensor.Tensor
import com.jayantkrish.jklol.training.LogFunction
import com.jayantkrish.jklol.training.NullLogFunction

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

  /** flatMap is the monad's bind operator. It chains two
    * probabilistic computations together in the natural way
    * where f represents a conditional distribution P(B | A).
    * Hence, binding f to a distribution P(A) returns the
    * marginal distribution over B, sum_a P(A=a) P(B | A=a).
    */
  def flatMap[B](f: A => Pp[B]): Pp[B] = BindPp(this, PpContinuationFunction(f))
  
  /** Implements a single search step of beam search.
   */
  def searchStep[C](env: Env, logProb: Double, continuation: PpContinuation[A,C],
    queue: PpSearchQueue[C], finished: PpSearchQueue[C]): Unit = {
    val v = step(env, logProb, queue.graph, queue.log)
    continuation.searchStep(v._1, v._2, v._3, queue, finished)
  }

  def lastSearchStep(env: Env, logProb: Double, queue: PpSearchQueue[A],
      finished: PpSearchQueue[A]): Unit = {
    val v = step(env, logProb, queue.graph, queue.log)
    finished.offer(ValuePp(v._1), v._2, v._3, v._2)
  }

  def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (A, Env, Double)

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

    val queue = new BeamPpSearchQueue[A](beamSize, stateCost, graph, log)
    val finished = new BeamPpSearchQueue[A](beamSize, stateCost, graph, log)

    val startEnv = env.setLog(log)
    queue.offer(this, env, 0.0, env)

    val beam = new Array[SearchState[A]](beamSize)
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
        val state = beam(i)
        state.value.lastSearchStep(state.env, state.logProb, queue, finished)
      }
    }

    // println(numIters)

    val finishedQueue = finished.queue
    val numFinished = finishedQueue.size
    val finishedItems = finishedQueue.getItems.slice(0, numFinished)
    val finishedScores = finishedQueue.getScores.slice(0, numFinished)

    val executions = finishedItems.zip(finishedScores).sortBy(x => -1 * x._2).map(
      x => new Execution(x._1.value.asInstanceOf[ValuePp[A]].value, x._1.env, x._2)
    )

    new PpBeamMarginals(executions.toSeq, queue.graph, numIters)
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

case class BindPp[A, C](b: Pp[C], f: PpContinuation[C, A]) extends Pp[A] {
  override def flatMap[B](g: A => Pp[B]) = BindPp(b, f.append(g))
  
  override def searchStep[C](env: Env, logProb: Double, continuation: PpContinuation[A,C],
    queue: PpSearchQueue[C], finished: PpSearchQueue[C]): Unit = {
    b.searchStep(env, logProb, f.append(continuation), queue, finished)
  }

  override def lastSearchStep(env: Env, logProb: Double, queue: PpSearchQueue[A],
      finished: PpSearchQueue[A]): Unit = {
    b.searchStep(env, logProb, f, queue, finished)
  }

  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (A, Env, Double) = {
    throw new UnsupportedOperationException("This method shouldn't ever get called.")
  }
}

/** Categorical distribution representing a nondeterministic
  * choice of an element of dist. The elements of dist are
  * scores, i.e., log probabilities.
  */
case class CategoricalPp[A](dist: Seq[(A, Double)]) extends Pp[A] {
  
  override def searchStep[C](env: Env, logProb: Double, continuation: PpContinuation[A,C],
    queue: PpSearchQueue[C], finished: PpSearchQueue[C]): Unit = {
    dist.foreach(x => queue.offer(BindPp(ValuePp(x._1), continuation), env, logProb + x._2, env))
  }

  override def lastSearchStep(env: Env, logProb: Double, queue: PpSearchQueue[A],
      finished: PpSearchQueue[A]): Unit = {
    dist.foreach(x => finished.offer(ValuePp(x._1), env, logProb + x._2, env))
  }

  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (A, Env, Double) = {
    throw new UnsupportedOperationException("This method shouldn't ever get called.")
  }
}

case class ValuePp[A](value: A) extends Pp[A] {
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (A, Env, Double) = {
    (value, env, logProb)
  }
}

case class ScorePp(score: Double) extends Pp[Unit] {
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (Unit, Env, Double) = {
    ((), env, logProb + Math.log(score))
  }
}

case class GetEnv() extends Pp[Env] {
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (Env, Env, Double) = {
    (env, env, logProb)
  }
}

case class SetEnv(nextEnv: Env) extends Pp[Unit] {
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (Unit, Env, Double) = {
    ((), nextEnv, logProb)
  }
}

case class SetVar(name: String, value: AnyRef) extends Pp[Unit] {
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (Unit, Env, Double) = {
    val nextEnv = env.setVar(name, value)
    ((), nextEnv, logProb)
  }
}

case class SetVarInt(nameInt: Int, value: AnyRef) extends Pp[Unit] {
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (Unit, Env, Double) = {
    val nextEnv = env.setVar(nameInt, value)
    ((), nextEnv, logProb)
  }
}

// Class for collapsing out multiple choices into a single choice
case class CollapsedSearch[A](dist: Pp[A]) extends Pp[A] {
  override def searchStep[B](env: Env, logProb: Double, continuation: PpContinuation[A, B],
    queue: PpSearchQueue[B], finished: PpSearchQueue[B]) = {
    val wrappedQueue = new ContinuationPpSearchQueue(queue, continuation)
    val nextQueue = new EnumeratePpSearchQueue[A](queue.stateCost, queue.graph, queue.log, wrappedQueue)
    
    dist.lastSearchStep(env, logProb, nextQueue, wrappedQueue)
  }
  
  override def lastSearchStep(env: Env, logProb: Double,
      queue: PpSearchQueue[A], finished: PpSearchQueue[A]) = {
    val nextQueue = new EnumeratePpSearchQueue[A](queue.stateCost, queue.graph, queue.log, finished)
    dist.lastSearchStep(env, logProb, nextQueue, finished)
  }
  
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (A, Env, Double) = {
    throw new UnsupportedOperationException("This method shouldn't ever get called.")
  }
}

// Classes for representing computation graph elements.

case class ParameterPp(name: String, id: Int) extends Pp[CompGraphNode] {
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction
      ): (CompGraphNode, Env, Double) = {
    val node = if (name == null) {
      graph.get(id)
    } else {
      graph.getParameter(name)
    }
    (node, env, logProb)
  }
}

case class ConstantTensorPp(tensor: Tensor) extends Pp[CompGraphNode] {
  
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction
      ): (CompGraphNode, Env, Double) = {
    val constNode = new ConstantNode(graph.nextId(), graph, tensor)
    graph.addNode(constNode)
    (constNode, env, logProb)
  }
}

case class ParameterizedCategoricalPp[A](items: Array[A], parameter: CompGraphNode,
    keyPrefix: Array[Int]) extends Pp[A] {

  def getTensor(): (Tensor, Long, Int) = {
    val paramTensor = parameter.value
    val startKeyNum = paramTensor.dimKeyPrefixToKeyNum(keyPrefix)
    val endKeyNum = if (keyPrefix.length == 0) {
      paramTensor.getMaxKeyNum
    } else {
      startKeyNum + (paramTensor.getDimensionOffsets())(keyPrefix.length - 1)
    }
    val numTensorValues = (endKeyNum - startKeyNum).asInstanceOf[Int]

    if (numTensorValues != items.length) {
      Preconditions.checkState(
          numTensorValues == items.length,
          "parameter dimensionality %s doesn't match item's %s (%s)",
          numTensorValues.asInstanceOf[AnyRef], items.length.asInstanceOf[AnyRef],
          Arrays.toString(items.asInstanceOf[Array[AnyRef]]))
    }

    (paramTensor, startKeyNum, numTensorValues) 
  }
  
  def makeLabelIndicator(keyNum: Long, params: Tensor): Tensor = {
    SparseTensor.singleElement(
      params.getDimensionNumbers,
      params.getDimensionSizes, params.keyNumToDimKey(keyNum), 1.0
    )
  }

  override def searchStep[B](env: Env, logProb: Double,
      continuation: PpContinuation[A, B], queue: PpSearchQueue[B], finished: PpSearchQueue[B]) = {
      
    val (paramTensor, startKeyNum, numTensorValues) = getTensor
    for (i <- 0 until numTensorValues) {
      val keyNum = startKeyNum + i
      val nextEnv = env.addLabel(parameter, makeLabelIndicator(keyNum, paramTensor))
      queue.offer(BindPp(ValuePp(items(i)), continuation), nextEnv, logProb + paramTensor.get(keyNum), env)
    }
  }
  
  override def lastSearchStep(env: Env, logProb: Double,
      queue: PpSearchQueue[A], finished: PpSearchQueue[A]) = {
      
    val (paramTensor, startKeyNum, numTensorValues) = getTensor
    for (i <- 0 until numTensorValues) {
      val keyNum = startKeyNum + i
      val nextEnv = env.addLabel(parameter, makeLabelIndicator(keyNum, paramTensor))
      finished.offer(ValuePp(items(i)), nextEnv, logProb + paramTensor.get(keyNum), env)
    }
  }
  
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (A, Env, Double) = {
      throw new UnsupportedOperationException("This method shouldn't ever get called.")
  }
}

case class ParameterizedArrayCategoricalPp[A](items: Array[A], parameters: Array[CompGraphNode]) extends Pp[A] {
  Preconditions.checkArgument(
    items.length == parameters.length,
    "Expected arrays to be equal length: items (%s) and parameters (%s)",
    items.length.asInstanceOf[AnyRef], parameters.length.asInstanceOf[AnyRef]
  )

  override def searchStep[B](env: Env, logProb: Double,
      continuation: PpContinuation[A, B], queue: PpSearchQueue[B], finished: PpSearchQueue[B]) = {
    val paramValues = parameters.map(x => {
      Preconditions.checkState(x.value.getDimensionNumbers.length == 0)
      x.value.get(0)
    })

    for (i <- 0 until items.length) {
      val nextEnv = env.addLabel(parameters(i), DenseTensor.scalar(1.0))
      queue.offer(BindPp(ValuePp(items(i)), continuation), nextEnv, logProb + paramValues(i), env)
    }
  }
  
  override def lastSearchStep(env: Env, logProb: Double,
      queue: PpSearchQueue[A], finished: PpSearchQueue[A]) = {
    val paramValues = parameters.map(x => {
      Preconditions.checkState(x.value.getDimensionNumbers.length == 0)
      x.value.get(0)
    })

    for (i <- 0 until items.length) {
      val nextEnv = env.addLabel(parameters(i), DenseTensor.scalar(1.0))
      finished.offer(ValuePp(items(i)), nextEnv, logProb + paramValues(i), env)
    }
  }
  
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (A, Env, Double) = {
    throw new UnsupportedOperationException("This method shouldn't ever get called.")
  }
}

case class StartTimerPp(timerName: String) extends Pp[Unit] {
  override def searchStep[B](env: Env, logProb: Double,
      continuation: PpContinuation[Unit, B], queue: PpSearchQueue[B], finished: PpSearchQueue[B]) = {
    queue.offer(BindPp(ValuePp(()), continuation), env.startTimer(timerName), logProb, env)
  }
  
  override def lastSearchStep(env: Env, logProb: Double,
      queue: PpSearchQueue[Unit], finished: PpSearchQueue[Unit]) = {
    queue.offer(ValuePp(()), env.startTimer(timerName), logProb, env)
  }
  
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (Unit, Env, Double) = {
    throw new UnsupportedOperationException("This method shouldn't ever get called.")
  }
}

case class StopTimerPp(timerName: String) extends Pp[Unit] {

  override def searchStep[B](env: Env, logProb: Double,
      continuation: PpContinuation[Unit, B], queue: PpSearchQueue[B], finished: PpSearchQueue[B]) = {
    queue.offer(BindPp(ValuePp(()), continuation), env.stopTimer(timerName), logProb, env)
  }
  
  override def lastSearchStep(env: Env, logProb: Double,
      queue: PpSearchQueue[Unit], finished: PpSearchQueue[Unit]) = {
    queue.offer(ValuePp(()), env.stopTimer(timerName), logProb, env)
  }
  
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (Unit, Env, Double) = {
    throw new UnsupportedOperationException("This method shouldn't ever get called.")
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
  
  def score(score: Double): Pp[Unit] = {
    choose(Seq(()), Seq(score))
  }

  // Methods for timing execution
  def startTimer(name: String): Pp[Unit] = {
    StartTimerPp(name)
  }

  def stopTimer(name: String): Pp[Unit] = {
    StopTimerPp(name)
  }
}