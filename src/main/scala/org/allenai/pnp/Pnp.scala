package org.allenai.pnp

import scala.collection.JavaConverters._
import com.google.common.base.Preconditions
import com.jayantkrish.jklol.training.LogFunction
import com.jayantkrish.jklol.training.NullLogFunction
import com.jayantkrish.jklol.util.CountAccumulator
import edu.cmu.dynet._
import scala.collection.mutable.MapBuilder

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
  def searchStep[C](env: Env, logProb: Double,
    context: PnpInferenceContext, continuation: PnpContinuation[A,C],
    queue: PnpSearchQueue[C], finished: PnpSearchQueue[C]): Unit

  /** Implements a single step of forward sampling.
    */
  def sampleStep[C](env: Env, logProb: Double,
    context: PnpInferenceContext, continuation: PnpContinuation[A,C],
    queue: PnpSearchQueue[C], finished: PnpSearchQueue[C]): Unit
    
  // Methods that do not need to be overriden

  def map[B](f: A => B): Pnp[B] = flatMap { a => Pnp.value(f(a)) }

  /** Performs a beam search over executions of this program returning
    * at most beamSize execution results. env is the initial global
    * state of the program, and graph is the initial computation graph.
    * These must contain values for any global variables or neural network
    * parameters referenced in the program.
    */
  def beamSearch(beamSize: Int = 1, maxIters: Int = -1, env: Env = Env.init,
      context: PnpInferenceContext = PnpInferenceContext.init): PnpBeamMarginals[A] = {

    val queue = new BeamPnpSearchQueue[A](beamSize)
    val finished = new BeamPnpSearchQueue[A](beamSize)
    
    val endContinuation = new PnpEndContinuation[A]()

    queue.offer(this, env, 0.0, context, null, null)

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
        state.value.searchStep(state.env, state.logProb, context, endContinuation, queue, finished)
        
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

    new PnpBeamMarginals(executions.toSeq, numIters)
  }

  // Version of beam search for programs that don't have trainable
  // parameters
  
  def beamSearch(k: Int, model: PnpModel): PnpBeamMarginals[A] = {
    ComputationGraph.renew()
    val context = PnpInferenceContext.init(model)
    beamSearch(k, Env.init, context)
  }

  def beamSearch(k: Int, env: Env): PnpBeamMarginals[A] = {
    beamSearch(k, env, PnpInferenceContext.init)
  }

  def beamSearch(k: Int, env: Env, context: PnpInferenceContext): PnpBeamMarginals[A] = {
    beamSearch(k, -1, env, context)
  }

  def inOneStep(): Pnp[A] = {
    CollapsedSearch(this)
  }
  
  def sample(numSamples: Int = 1, env: Env = Env.init, context: PnpInferenceContext = PnpInferenceContext.init): Execution[A] = {

    // TODO: make the cost interact with the sampling.
    val queue = new BeamPnpSearchQueue[A](1)
    val finished = new BeamPnpSearchQueue[A](1)
    
    val endContinuation = new PnpEndContinuation[A]()
    
    sampleStep(env, 0.0, context, endContinuation, queue, finished)
    
    val numFinished = finished.queue.size
    val finishedItems = finished.queue.getItems.slice(0, numFinished)
    val finishedScores = finished.queue.getScores.slice(0, numFinished)

    val executions = finishedItems.zip(finishedScores).sortBy(x => -1 * x._2).map(
      x => new Execution(x._1.value.asInstanceOf[ValuePnp[A]].value, x._1.env, x._2)
    )

    Preconditions.checkState(executions.size == 1)
    executions(0)
  }
}

case class BindPnp[A, C](b: Pnp[C], f: PnpContinuation[C, A]) extends Pnp[A] {
  override def flatMap[B](g: A => Pnp[B]) = BindPnp(b, f.append(g))
  
  override def searchStep[D](env: Env, logProb: Double, context: PnpInferenceContext,
    continuation: PnpContinuation[A,D], queue: PnpSearchQueue[D], finished: PnpSearchQueue[D]): Unit = {
    b.searchStep(env, logProb, context, f.append(continuation), queue, finished)
  }
  
  override def sampleStep[D](env: Env, logProb: Double, context: PnpInferenceContext,
    continuation: PnpContinuation[A,D], queue: PnpSearchQueue[D], finished: PnpSearchQueue[D]): Unit = {
    b.sampleStep(env, logProb, context, f.append(continuation), queue, finished)
  }
}

/** Categorical distribution representing a nondeterministic
  * choice of an element of dist. The elements of dist are
  * scores, i.e., log probabilities.
  */
case class CategoricalPnp[A](dist: Array[(A, Double)], tag: Any) extends Pnp[A] {
  
  override def searchStep[C](env: Env, logProb: Double, context: PnpInferenceContext,
    continuation: PnpContinuation[A,C], queue: PnpSearchQueue[C], finished: PnpSearchQueue[C]): Unit = {
    dist.foreach {
      case (value, valueLogProb) => {
        val newPnp = BindPnp(ValuePnp(value), continuation)
        queue.offer(newPnp, env, logProb + valueLogProb, context, tag, value)
      }
    }
  }
  
  override def sampleStep[C](env: Env, logProb: Double, context: PnpInferenceContext, continuation: PnpContinuation[A,C],
    queue: PnpSearchQueue[C], finished: PnpSearchQueue[C]): Unit = {
    // TODO: This code assumes that the distribution is locally normalized.
    val expDist = dist.map {
      case (value, valueLogProb) => {
        val score = valueLogProb + context.computeScore(tag, value, env)
        (value, Math.exp(score))
      }
    }
    val totalProb = expDist.map(_._2).sum
    val draw = Math.random() * totalProb
    
    var s = 0.0
    var choice = -1 
    for (i <- 0 until expDist.length) {
      s += expDist(i)._2
      if (draw <= s && choice == -1) {
        choice = i
      }
    }
    
    val (value, choiceLogProb) = dist(choice)
    // TODO (check me)
    val newLogProb = logProb + choiceLogProb
    ValuePnp(value).sampleStep(env, newLogProb, context, continuation, queue, finished)
  }
}

case class ParameterizedCategoricalPnp[A](items: Array[A], parameter: Expression, tag: Any) extends Pnp[A] {

  def getTensor(graph: CompGraph): (Tensor, Int) = {
    val paramTensor = if (graph.locallyNormalized) {
      val softmaxed = Expression.logSoftmax(parameter)
      ComputationGraph.incrementalForward(softmaxed)
    } else {
      ComputationGraph.incrementalForward(parameter)
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
      ComputationGraph.printGraphViz()
      println(paramTensor.toVector.size)

      Preconditions.checkState(
          size == items.length,
          "parameter dimensionality %s doesn't match item's %s (%s) at tag %s",
          dims.size.asInstanceOf[AnyRef], items.length.asInstanceOf[AnyRef],
          items, tag.toString)
    }

    (paramTensor, size) 
  }

  override def searchStep[B](env: Env, logProb: Double, context: PnpInferenceContext,
      continuation: PnpContinuation[A, B], queue: PnpSearchQueue[B], finished: PnpSearchQueue[B]) = {
      
    val (paramTensor, numTensorValues) = getTensor(context.compGraph)
    val v = paramTensor.toVector
    for (i <- 0 until numTensorValues) {
      val nextEnv = env.addLabel(parameter, i)
      val nextLogProb = logProb + v(i)
      queue.offer(BindPnp(ValuePnp(items(i)), continuation), nextEnv, nextLogProb, context, tag, items(i))
    }
  }
  
  override def sampleStep[D](env: Env, logProb: Double, context: PnpInferenceContext, continuation: PnpContinuation[A,D],
    queue: PnpSearchQueue[D], finished: PnpSearchQueue[D]): Unit = {
    
    val (paramTensor, numTensorValues) = getTensor(context.compGraph)
    val logScores = paramTensor.toSeq.toArray

    // TODO: This code assumes that the distribution is locally normalized.
    val scores = items.zip(logScores).map {
      case (value, valueLogProb) => {
        // Add in the state cost
        valueLogProb + context.computeScore(tag, value, env)
      }
    }.map(Math.exp(_))

    val totalProb = scores.sum
    val draw = Math.random() * totalProb

    var s = 0.0
    var choice = -1 
    for (i <- 0 until scores.length) {
      s += scores(i)
      if (draw <= s && choice == -1) {
        choice = i
      }
    }

    val value = items(choice)
    val choiceLogProb = scores(choice)
    val nextEnv = env.addLabel(parameter, choice)
    ValuePnp(value).sampleStep(nextEnv, logProb + choiceLogProb, context, continuation, queue, finished)
  }
}

case class ScorePnp(score: Double) extends Pnp[Unit] {
  override def searchStep[C](env: Env, logProb: Double, context: PnpInferenceContext, continuation: PnpContinuation[Unit,C],
    queue: PnpSearchQueue[C], finished: PnpSearchQueue[C]): Unit = {
    // TODO(joelgrus) should we be taking log here?
    val nextLogProb = logProb + Math.log(score)
    continuation.searchStep((), env, nextLogProb, context, queue, finished)
  }
  
  override def sampleStep[C](env: Env, logProb: Double, context: PnpInferenceContext, continuation: PnpContinuation[Unit,C],
    queue: PnpSearchQueue[C], finished: PnpSearchQueue[C]): Unit = {
    throw new UnsupportedOperationException("Sampling with score functions is not implemented.")
  }
}

// Class for collapsing out multiple choices into a single choice
case class CollapsedSearch[A](dist: Pnp[A]) extends Pnp[A] {
  override def searchStep[B](env: Env, logProb: Double, context: PnpInferenceContext, continuation: PnpContinuation[A, B],
    queue: PnpSearchQueue[B], finished: PnpSearchQueue[B]) = {
    val wrappedQueue = new ContinuationPnpSearchQueue(queue, continuation)
    val nextQueue = new EnumeratePnpSearchQueue[A](wrappedQueue)
    val endContinuation = new PnpEndContinuation[A]()

    dist.searchStep(env, logProb, context, endContinuation, nextQueue, wrappedQueue)
  }
  
  override def sampleStep[B](env: Env, logProb: Double, context: PnpInferenceContext, continuation: PnpContinuation[A,B],
    queue: PnpSearchQueue[B], finished: PnpSearchQueue[B]): Unit = {
    dist.sampleStep(env, logProb, context, continuation, queue, finished)
  }
}

case class ValuePnp[A](value: A) extends Pnp[A] {
  override def searchStep[C](env: Env, logProb: Double, context: PnpInferenceContext, continuation: PnpContinuation[A,C],
    queue: PnpSearchQueue[C], finished: PnpSearchQueue[C]): Unit = {
    continuation.searchStep(value, env, logProb, context, queue, finished)
  }
  
  override def sampleStep[C](env: Env, logProb: Double, context: PnpInferenceContext, continuation: PnpContinuation[A,C],
    queue: PnpSearchQueue[C], finished: PnpSearchQueue[C]): Unit = {
    continuation.sampleStep(value, env, logProb, context, queue, finished)
  }
}

case class GetEnv() extends Pnp[Env] {
  override def searchStep[C](env: Env, logProb: Double, context: PnpInferenceContext, continuation: PnpContinuation[Env,C],
    queue: PnpSearchQueue[C], finished: PnpSearchQueue[C]): Unit = {
    continuation.searchStep(env, env, logProb, context, queue, finished)
  }
  
  override def sampleStep[C](env: Env, logProb: Double, context: PnpInferenceContext, continuation: PnpContinuation[Env,C],
    queue: PnpSearchQueue[C], finished: PnpSearchQueue[C]): Unit = {
    continuation.sampleStep(env, env, logProb, context, queue, finished)
  }
}

case class SetEnv(nextEnv: Env) extends Pnp[Unit] {
  override def searchStep[C](env: Env, logProb: Double, context: PnpInferenceContext, continuation: PnpContinuation[Unit,C],
    queue: PnpSearchQueue[C], finished: PnpSearchQueue[C]): Unit = {
    continuation.searchStep((), nextEnv, logProb, context, queue, finished)
  }
  
  override def sampleStep[C](env: Env, logProb: Double, context: PnpInferenceContext, continuation: PnpContinuation[Unit,C],
    queue: PnpSearchQueue[C], finished: PnpSearchQueue[C]): Unit = {
    continuation.sampleStep((), nextEnv, logProb, context, queue, finished)
  }
}

// Classes for representing computation graph elements.
case class ComputationGraphPnp() extends Pnp[CompGraph] {
  override def searchStep[C](env: Env, logProb: Double, context: PnpInferenceContext, continuation: PnpContinuation[CompGraph,C],
    queue: PnpSearchQueue[C], finished: PnpSearchQueue[C]): Unit = {
    continuation.searchStep(context.compGraph, env, logProb, context, queue, finished)
  }
  
  override def sampleStep[C](env: Env, logProb: Double, context: PnpInferenceContext, continuation: PnpContinuation[CompGraph,C],
    queue: PnpSearchQueue[C], finished: PnpSearchQueue[C]): Unit = {
    continuation.sampleStep(context.compGraph, env, logProb, context, queue, finished)
  }
}

case class StartTimerPnp(timerName: String) extends Pnp[Unit] {
  override def searchStep[B](env: Env, logProb: Double, context: PnpInferenceContext, continuation: PnpContinuation[Unit, B],
      queue: PnpSearchQueue[B], finished: PnpSearchQueue[B]) = {
    queue.offer(BindPnp(ValuePnp(()), continuation), env, logProb, context, null, null)
  }
  
  override def sampleStep[B](env: Env, logProb: Double, context: PnpInferenceContext, continuation: PnpContinuation[Unit, B],
      queue: PnpSearchQueue[B], finished: PnpSearchQueue[B]) = {
    // TODO: figure out how timers should work with sampling.
    continuation.sampleStep((), env, logProb, context, queue, finished)
  }
}

case class StopTimerPnp(timerName: String) extends Pnp[Unit] {
  override def searchStep[B](env: Env, logProb: Double, context: PnpInferenceContext,
      continuation: PnpContinuation[Unit, B], queue: PnpSearchQueue[B], finished: PnpSearchQueue[B]) = {
    queue.offer(BindPnp(ValuePnp(()), continuation), env, logProb, context, null, null)
  }
  
  override def sampleStep[B](env: Env, logProb: Double, context: PnpInferenceContext, continuation: PnpContinuation[Unit, B],
      queue: PnpSearchQueue[B], finished: PnpSearchQueue[B]) = {
    // TODO: figure out how timers should work with sampling.
    continuation.sampleStep((), env, logProb, context, queue, finished)
  }
}

class Execution[A](val value: A, val env: Env, val logProb: Double) {
  def prob = Math.exp(logProb)

  override def toString: String = {
    "[Execution " + value + " " + logProb + "]"
  }
}

class PnpBeamMarginals[A](val executions: Seq[Execution[A]], val searchSteps: Int) {

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
    // .getCountMap.asScala.map(x => (x._1, x._2.doubleValue())).toMap
  }

  def condition(pred: (A, Env) => Boolean): PnpBeamMarginals[A] = {
    return new PnpBeamMarginals(executions.filter(x => pred(x.value, x.env)), searchSteps)
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
    CategoricalPnp(dist.map(x => (x._1, Math.log(x._2))).toArray, null)
  }

  def choose[A](items: Seq[A], weights: Seq[Double]): Pnp[A] = {
    CategoricalPnp(items.zip(weights).map(x => (x._1, Math.log(x._2))).toArray, null)
  }
  
  def choose[A](items: Seq[A]): Pnp[A] = {
    CategoricalPnp(items.map(x => (x, 0.0)).toArray, null)
  }
  
  def chooseTag[A](items: Seq[A], tag: Any): Pnp[A] = {
    CategoricalPnp(items.map(x => (x, 0.0)).toArray, tag)
  }

  /** The failure program that has no executions.
    */
  def fail[A]: Pnp[A] = { CategoricalPnp(Array.empty[(A, Double)], null) }

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
    for {
      env <- getEnv()
      nextEnv = env.setVar(name, value)
      result <- setEnv(nextEnv)
    } yield {
      result
    }
  }

  def setVar[A <: AnyRef](nameInt: Int, value: A): Pnp[Unit] = {
    for {
      env <- getEnv()
      nextEnv = env.setVar(nameInt, value)
      result <- setEnv(nextEnv)
    } yield {
      result
    }
  }

  def isVarBound(name: String): Pnp[Boolean] = {
    for {
      x <- getEnv
    } yield {
      x.isVarBound(name)
    }
  }

  // Methods for interacting with the computation graph
  
  /** Get the computation graph
    */
  def computationGraph(): Pnp[CompGraph] = { ComputationGraphPnp() }

  /** Get a neural network parameter by name.
    */
  def param(name: String): Pnp[Expression] = {
    for {
      cg <- Pnp.computationGraph()
    } yield {
      Expression.parameter(cg.getParameter(name))
    }
  }

  /** Add a FloatVector to the computation graph as a constant.
    */
  def constant(dims: Dim, vector: FloatVector): Pnp[Expression] = {
    for { _ <- ValuePnp(()) } yield Expression.input(dims, vector)
  }

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