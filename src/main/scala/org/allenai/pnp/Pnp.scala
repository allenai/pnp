package org.allenai.pnp

import java.util.Arrays

import com.google.common.base.Preconditions
import com.jayantkrish.jklol.training.LogFunction
import com.jayantkrish.jklol.training.NullLogFunction

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._
import com.jayantkrish.jklol.util.CountAccumulator


/** Probabilistic neural program monad. Pnp[X] represents a
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
  * Probabilistic neural programs are constructed and manipulated
  * using for/yield comprehensions and the functions in the
  * Pnp object.
  */
trait Pnp[A] {

  /** flatMap is the monad's bind operator. It chains two
    * probabilistic computations together in the natural way
    * where f represents a conditional distribution P(B | A).
    * Hence, binding f to a distribution P(A) returns the
    * marginal distribution over B, sum_a P(A=a) P(B | A=a).
    */
  def flatMap[B](f: A => Pnp[B]): Pnp[B] = BindPnp(this, PnpContinuationFunction(f))
  
  /** Implements a single search step of beam search.
   */
  def searchStep[C](env: Env, logProb: Double, continuation: PnpContinuation[A,C],
    queue: PnpSearchQueue[C], finished: PnpSearchQueue[C]): Unit = {
    val v = step(env, logProb, queue.graph, queue.log)
    continuation.searchStep(v._1, v._2, v._3, queue, finished)
  }

  def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (A, Env, Double)

  // Methods that do not need to be overriden

  def map[B](f: A => B): Pnp[B] = flatMap { a => Pnp.value(f(a)) }

  /** Performs a beam search over executions of this program returning
    * at most beamSize execution results. env is the initial global
    * state of the program, and graph is the initial computation graph.
    * These must contain values for any global variables or neural network
    * parameters referenced in the program.
    */
  def beamSearch(beamSize: Int, maxIters: Int, env: Env, stateCostArg: ExecutionScore,
    graph: CompGraph, log: LogFunction): PnpBeamMarginals[A] = {

    val stateCost = if (stateCostArg == null) {
      ExecutionScore.zero
    } else {
      stateCostArg
    }

    val queue = new BeamPnpSearchQueue[A](beamSize, stateCost, graph, log)
    val finished = new BeamPnpSearchQueue[A](beamSize, stateCost, graph, log)
    
    val endContinuation = new PnpEndContinuation[A]()

    val startEnv = env.setLog(log)
    queue.offer(this, env, 0.0, null, null, env)

    val beam = new Array[SearchState[A]](beamSize)
    var numIters = 0
    while (queue.queue.size > 0 && (maxIters < 0 || numIters < maxIters)) {
      numIters += 1
      // println(numIters + " " + queue.queue.size)

      val beamSize = queue.queue.size
      Array.copy(queue.queue.getItems, 0, beam, 0, beamSize)
      queue.queue.clear

      for (i <- 0 until beamSize) {
        val state = beam(i)
        // println(state.value)
        state.value.searchStep(state.env, state.logProb, endContinuation, queue, finished)
        
        // state.value.lastSearchStep(state.env, state.logProb, queue, finished)
      }
    }

    // println(numIters)

    val finishedQueue = finished.queue
    val numFinished = finishedQueue.size
    val finishedItems = finishedQueue.getItems.slice(0, numFinished)
    val finishedScores = finishedQueue.getScores.slice(0, numFinished)

    val executions = finishedItems.zip(finishedScores).sortBy(x => -1 * x._2).map(
      x => new Execution(x._1.value.asInstanceOf[ValuePnp[A]].value, x._1.env, x._2)
    )

    new PnpBeamMarginals(executions.toSeq, queue.graph, numIters)
  }

  def beamSearchWithFilter(beamSize: Int, env: Env, keepState: Env => Boolean,
    graph: CompGraph, log: LogFunction): PnpBeamMarginals[A] = {
    val cost = ExecutionScore.fromFilter(keepState)
    beamSearch(beamSize, -1, env, cost, graph, log)
  }

  // Version of beam search for programs that don't have trainable
  // parameters
  def beamSearch(k: Int): Seq[(A, Double)] = {
    beamSearch(k, Env.init).executions.map(x => (x.value, x.prob))
  }

  def beamSearch(k: Int, env: Env): PnpBeamMarginals[A] = {
    beamSearchWithFilter(k, env, (x: Env) => true)
  }

  def beamSearch(k: Int, env: Env, cg: CompGraph): PnpBeamMarginals[A] = {
    beamSearch(k, -1, env, ExecutionScore.zero, cg, new NullLogFunction())
  }

  def beamSearchWithFilter(
    k: Int, env: Env, keepState: Env => Boolean, cg: CompGraph
  ): PnpBeamMarginals[A] = {
    beamSearchWithFilter(k, env, keepState, cg, new NullLogFunction())
  }

  def beamSearchWithFilter(k: Int, env: Env, keepState: Env => Boolean): PnpBeamMarginals[A] = {
    beamSearchWithFilter(k, env, keepState, null, new NullLogFunction())
  }

  def inOneStep(): Pnp[A] = {
    CollapsedSearch(this)
  }
  
  def sample(k: Int, env: Env, stateCostArg: ExecutionScore,
    graph: CompGraph, log: LogFunction) = {
    
  }
}

case class BindPnp[A, C](b: Pnp[C], f: PnpContinuation[C, A]) extends Pnp[A] {
  override def flatMap[B](g: A => Pnp[B]) = BindPnp(b, f.append(g))
  
  override def searchStep[D](env: Env, logProb: Double, continuation: PnpContinuation[A,D],
    queue: PnpSearchQueue[D], finished: PnpSearchQueue[D]): Unit = {
    b.searchStep(env, logProb, f.append(continuation), queue, finished)
  }

  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (A, Env, Double) = {
    throw new UnsupportedOperationException("This method shouldn't ever get called.")
  }
}

/** Categorical distribution representing a nondeterministic
  * choice of an element of dist. The elements of dist are
  * scores, i.e., log probabilities.
  */
case class CategoricalPnp[A](dist: Seq[(A, Double)], tag: Any) extends Pnp[A] {
  
  override def searchStep[C](env: Env, logProb: Double, continuation: PnpContinuation[A,C],
    queue: PnpSearchQueue[C], finished: PnpSearchQueue[C]): Unit = {
    dist.foreach(x => queue.offer(BindPnp(ValuePnp(x._1), continuation), env, logProb + x._2,
        tag, x._1, env))
  }

  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (A, Env, Double) = {
    throw new UnsupportedOperationException("This method shouldn't ever get called.")
  }
}

case class ValuePnp[A](value: A) extends Pnp[A] {
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (A, Env, Double) = {
    (value, env, logProb)
  }
}

case class ScorePnp(score: Double) extends Pnp[Unit] {
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (Unit, Env, Double) = {
    ((), env, logProb + Math.log(score))
  }
}

case class GetEnv() extends Pnp[Env] {
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (Env, Env, Double) = {
    (env, env, logProb)
  }
}

case class SetEnv(nextEnv: Env) extends Pnp[Unit] {
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (Unit, Env, Double) = {
    ((), nextEnv, logProb)
  }
}

case class SetVar(name: String, value: AnyRef) extends Pnp[Unit] {
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (Unit, Env, Double) = {
    val nextEnv = env.setVar(name, value)
    ((), nextEnv, logProb)
  }
}

case class SetVarInt(nameInt: Int, value: AnyRef) extends Pnp[Unit] {
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (Unit, Env, Double) = {
    val nextEnv = env.setVar(nameInt, value)
    ((), nextEnv, logProb)
  }
}

// Class for collapsing out multiple choices into a single choice
case class CollapsedSearch[A](dist: Pnp[A]) extends Pnp[A] {
  override def searchStep[B](env: Env, logProb: Double, continuation: PnpContinuation[A, B],
    queue: PnpSearchQueue[B], finished: PnpSearchQueue[B]) = {
    val wrappedQueue = new ContinuationPnpSearchQueue(queue, continuation)
    val nextQueue = new EnumeratePnpSearchQueue[A](queue.stateCost, queue.graph, queue.log, wrappedQueue)
    val endContinuation = new PnpEndContinuation[A]()

    dist.searchStep(env, logProb, endContinuation, nextQueue, wrappedQueue)
  }
    
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (A, Env, Double) = {
    throw new UnsupportedOperationException("This method shouldn't ever get called.")
  }
}

// Classes for representing computation graph elements.

case class ParameterPnp(name: String) extends Pnp[Expression] {
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction
      ): (Expression, Env, Double) = {
    val expression = parameter(graph.cg, graph.getParameter(name))
    (expression, env, logProb)
  }
}

case class ComputationGraphPnp() extends Pnp[CompGraph] {
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction
      ): (CompGraph, Env, Double) = {
    (graph, env, logProb)
  }
}

case class FloatVectorPnp(dims: Dim, vector: FloatVector) extends Pnp[Expression] {
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction
      ): (Expression, Env, Double) = {
    val constNode = input(graph.cg, dims, vector)
    (constNode, env, logProb)
  }
}

case class ParameterizedCategoricalPnp[A](items: Array[A], parameter: Expression,
    tag: Any) extends Pnp[A] {

  def getTensor(graph: CompGraph): (Tensor, Int) = {
    val paramTensor = if (graph.locallyNormalized) {
      val softmaxed = log_softmax(parameter)
      graph.cg.incremental_forward(softmaxed)
    } else {
      graph.cg.incremental_forward(parameter)
    }

    val dims = paramTensor.getD
    
    val size = dims.size.asInstanceOf[Int]

    // Print the log (unnormalized) probability associated with each choice.
    /*
    for (i <- 0 until size) {
      println(as_vector(paramTensor).get(i))
    }
    */
    
    if (size != items.length) {
      graph.cg.print_graphviz()
      println(as_vector(paramTensor).size())

      Preconditions.checkState(
          size == items.length,
          "parameter dimensionality %s doesn't match item's %s (%s) at tag %s",
          dims.size.asInstanceOf[AnyRef], items.length.asInstanceOf[AnyRef],
          items, tag.toString)
    }

    (paramTensor, size) 
  }

  override def searchStep[B](env: Env, logProb: Double,
      continuation: PnpContinuation[A, B], queue: PnpSearchQueue[B], finished: PnpSearchQueue[B]) = {
      
    val (paramTensor, numTensorValues) = getTensor(queue.graph)
    val v = as_vector(paramTensor)
    for (i <- 0 until numTensorValues) {
      val nextEnv = env.addLabel(parameter, i)
      queue.offer(BindPnp(ValuePnp(items(i)), continuation), nextEnv, logProb + v.get(i),
          tag, items(i), env)
    }
  }

  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (A, Env, Double) = {
      throw new UnsupportedOperationException("This method shouldn't ever get called.")
  }
}

case class StartTimerPnp(timerName: String) extends Pnp[Unit] {
  override def searchStep[B](env: Env, logProb: Double,
      continuation: PnpContinuation[Unit, B], queue: PnpSearchQueue[B], finished: PnpSearchQueue[B]) = {
    queue.offer(BindPnp(ValuePnp(()), continuation), env.startTimer(timerName), logProb, null, null, env)
  }
  
  override def step(env: Env, logProb: Double, graph: CompGraph, log: LogFunction): (Unit, Env, Double) = {
    throw new UnsupportedOperationException("This method shouldn't ever get called.")
  }
}

case class StopTimerPnp(timerName: String) extends Pnp[Unit] {

  override def searchStep[B](env: Env, logProb: Double,
      continuation: PnpContinuation[Unit, B], queue: PnpSearchQueue[B], finished: PnpSearchQueue[B]) = {
    queue.offer(BindPnp(ValuePnp(()), continuation), env.stopTimer(timerName), logProb, null, null, env)
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

class PnpBeamMarginals[A](val executions: Seq[Execution[A]], val graph: CompGraph,
    val searchSteps: Int) {

  def logPartitionFunction(): Double = {
    if (executions.length > 0) {
      val logProbs = executions.map(x => x.logProb)
      val max = logProbs.max
      max + Math.log(logProbs.map(x => Math.exp(x - max)).sum)
    } else {
      Double.NegativeInfinity
    }
  }

  def partitionFunction(): Double = {
    executions.map(x => x.prob).sum
  }

  def marginals(): CountAccumulator[A] = {
    val counts = CountAccumulator.create[A]
    
    if (executions.length > 0) {
      val lpf = logPartitionFunction
      for (ex <- executions) {
        counts.increment(ex.value, Math.exp(ex.logProb - lpf))
      }
    }

    counts
  }

  def condition(pred: (A, Env) => Boolean): PnpBeamMarginals[A] = {
    return new PnpBeamMarginals(executions.filter(x => pred(x.value, x.env)), graph, searchSteps)
  }
}

object Pnp {

  /** Create a program that returns {@code value}
    */
  def value[A](value: A): Pnp[A] = { ValuePnp(value) }

  /** A nondeterministic choice. Creates a program
    * that chooses and returns a single value from
    * {@code dist} with the given probability.
    */
  def chooseMap[A](dist: Seq[(A, Double)]): Pnp[A] = {
    CategoricalPnp(dist.map(x => (x._1, Math.log(x._2))), null)
  }

  def choose[A](items: Seq[A], weights: Seq[Double]): Pnp[A] = {
    CategoricalPnp(items.zip(weights).map(x => (x._1, Math.log(x._2))), null)
  }
  
  def choose[A](items: Seq[A]): Pnp[A] = {
    CategoricalPnp(items.map(x => (x, 0.0)), null)
  }
  
  def chooseTag[A](items: Seq[A], tag: Any): Pnp[A] = {
    CategoricalPnp(items.map(x => (x, 0.0)), tag)
  }

  /** The failure program that has no executions.
    */
  def fail[A]: Pnp[A] = { CategoricalPnp(Seq.empty[(A, Double)], null) }

  def require(value: Boolean): Pnp[Unit] = {
    if (value) {
      Pnp.value(())
    } else {
      Pnp.fail
    }
  }

  // Methods for manipulating global program state

  /** Gets the environment (mutable state) of the program.
    * See the {@code getVar} method to get the value of a
    * single variable.
    */
  private def getEnv(): Pnp[Env] = { GetEnv() }

  /** Sets the environment (mutable state) of the program.
    * See the {@code setVar} method to set the value of a
    * single variable.
    */
  private def setEnv(env: Env): Pnp[Unit] = { SetEnv(env) }

  /** Gets the value of the variable {@code name} in the
    * program's environment.
    */
  def getVar[A](name: String): Pnp[A] = {
    for {
      x <- getEnv
    } yield {
      x.getVar[A](name)
    }
  }

  def getVar[A](nameInt: Int): Pnp[A] = {
    for {
      x <- getEnv
    } yield {
      x.getVar[A](nameInt)
    }
  }

  /** Sets the value of the variable {@code name} to
    * {@code value} in the program's environment.
    */
  def setVar[A <: AnyRef](name: String, value: A): Pnp[Unit] = {
    SetVar(name, value)
  }

  def setVar[A <: AnyRef](nameInt: Int, value: A): Pnp[Unit] = {
    SetVarInt(nameInt, value)
  }

  def isVarBound(name: String): Pnp[Boolean] = {
    for {
      x <- getEnv
    } yield {
      x.isVarBound(name)
    }
  }

  // Methods for interacting with the computation graph
  def computationGraph(): Pnp[CompGraph] = { ComputationGraphPnp() }

  /** Get a neural network parameter by name.
    */
  def param(name: String): Pnp[Expression] = { ParameterPnp(name) }

  /** Add a FloatVector to the computation graph as a constant.
    */
  def constant(dims: Dim, vector: FloatVector): Pnp[Expression] = { FloatVectorPnp(dims, vector) }

  /** Chooses an item. The ith item's score is the
    * ith index in parameter.
    */
  def choose[A](items: Array[A], parameter: Expression, tag: Any): Pnp[A] = {
    ParameterizedCategoricalPnp(items, parameter, tag)
  }
  
  def choose[A](items: Array[A], parameter: Expression): Pnp[A] = {
    choose(items, parameter, null)
  }

  /** Add a scalar value to the score of this execution.
    */
  def score(parameter: Expression): Pnp[Unit] = {
    ParameterizedCategoricalPnp(Array(()), parameter, null)
  }
  
  def score(score: Double): Pnp[Unit] = {
    choose(Seq(()), Seq(score))
  }

  // Methods for timing execution
  def startTimer(name: String): Pnp[Unit] = {
    StartTimerPnp(name)
  }

  def stopTimer(name: String): Pnp[Unit] = {
    StopTimerPnp(name)
  }
}