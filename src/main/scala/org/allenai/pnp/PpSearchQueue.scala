package org.allenai.pnp

import com.jayantkrish.jklol.training.LogFunction
import com.jayantkrish.jklol.util.KbestQueue

trait PpSearchQueue[A] {
  val graph: CompGraph
  val stateCost: ExecutionScore
  val log: LogFunction

  def offer(value: Pp[A], env: Env, logProb: Double, tag: Any, choice: Any, myEnv: Env): Unit
}

class BeamPpSearchQueue[A](size: Int, val stateCost: ExecutionScore,
    val graph: CompGraph, val log: LogFunction) extends PpSearchQueue[A] {

  val queue = new KbestQueue(size, Array.empty[SearchState[A]])

  override def offer(value: Pp[A], env: Env, logProb: Double, tag: Any,
      choice: Any, myEnv: Env): Unit = {
    val stateLogProb = stateCost(tag, choice, env) + logProb
    if (stateLogProb > Double.NegativeInfinity) {
      queue.offer(SearchState(value, env, stateLogProb, tag, choice), logProb)
    }
  }
}

class EnumeratePpSearchQueue[A] (
    val stateCost: ExecutionScore,
    val graph: CompGraph, val log: LogFunction,
    val finished: PpSearchQueue[A]
) extends PpSearchQueue[A] {
  override def offer(value: Pp[A], env: Env, logProb: Double, tag: Any,
      choice: Any, myEnv: Env): Unit = {
    myEnv.pauseTimers()
    val stateLogProb = stateCost(tag, choice, env) + logProb
    if (stateLogProb > Double.NegativeInfinity) {
      env.resumeTimers()
      value.lastSearchStep(env, logProb, this, finished)
      env.pauseTimers()
    }
    myEnv.resumeTimers()
  }
}

class ContinuationPpSearchQueue[A, B] (
    val queue: PpSearchQueue[B],
    val cont: PpContinuation[A,B]
) extends PpSearchQueue[A] {
  
  val graph = queue.graph
  val stateCost = queue.stateCost
  val log = queue.log
  
  override def offer(value: Pp[A], env: Env, logProb: Double, tag: Any,
      choice: Any, myEnv: Env): Unit = {
    queue.offer(BindPp(value, cont), env, logProb, tag, choice, myEnv)
  }
}

case class SearchState[A](val value: Pp[A], val env: Env, val logProb: Double,
    val tag: Any, val choice: Any) {
}
