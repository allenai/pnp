package org.allenai.pnp

import com.jayantkrish.jklol.training.LogFunction
import com.jayantkrish.jklol.util.KbestQueue

trait PnpSearchQueue[A] {
  val graph: CompGraph
  val log: LogFunction

  def offer(value: Pnp[A], env: Env, logProb: Double, tag: Any, choice: Any, myEnv: Env): Unit
}

class BeamPnpSearchQueue[A](size: Int, val graph: CompGraph, val log: LogFunction) extends PnpSearchQueue[A] {

  val queue = new KbestQueue(size, Array.empty[SearchState[A]])

  override def offer(value: Pnp[A], env: Env, logProb: Double, tag: Any,
      choice: Any, myEnv: Env): Unit = {
    val stateLogProb = Pnp.stateCost(tag, choice, env) + logProb
    if (stateLogProb > Double.NegativeInfinity) {
      queue.offer(SearchState(value, env, stateLogProb, tag, choice), stateLogProb)
    }
  }
}

class EnumeratePnpSearchQueue[A] (
    val graph: CompGraph, val log: LogFunction,
    val finished: PnpSearchQueue[A]
) extends PnpSearchQueue[A] {
  val endContinuation = new PnpEndContinuation[A]
  
  override def offer(value: Pnp[A], env: Env, logProb: Double, tag: Any,
      choice: Any, myEnv: Env): Unit = {
    myEnv.pauseTimers()
    val stateLogProb = Pnp.stateCost(tag, choice, env) + logProb
    if (stateLogProb > Double.NegativeInfinity) {
      env.resumeTimers()
      value.searchStep(env, stateLogProb, endContinuation, this, finished)
      env.pauseTimers()
    }
    myEnv.resumeTimers()
  }
}

class ContinuationPnpSearchQueue[A, B] (
    val queue: PnpSearchQueue[B],
    val cont: PnpContinuation[A,B]
) extends PnpSearchQueue[A] {
  
  val graph = queue.graph
  val log = queue.log
  
  override def offer(value: Pnp[A], env: Env, logProb: Double, tag: Any,
      choice: Any, myEnv: Env): Unit = {
    queue.offer(BindPnp(value, cont), env, logProb, tag, choice, myEnv)
  }
}

case class SearchState[A](val value: Pnp[A], val env: Env, val logProb: Double,
    val tag: Any, val choice: Any) {
}
