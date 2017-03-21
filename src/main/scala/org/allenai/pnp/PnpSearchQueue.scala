package org.allenai.pnp

import com.jayantkrish.jklol.training.LogFunction
import com.jayantkrish.jklol.util.KbestQueue

trait PnpSearchQueue[A] {
  def offer(value: Pnp[A], env: Env, logProb: Double, context: PnpInferenceContext, tag: Any, choice: Any): Unit
}

class BeamPnpSearchQueue[A](size: Int) extends PnpSearchQueue[A] {

  val queue = new KbestQueue(size, Array.empty[SearchState[A]])

  override def offer(value: Pnp[A], env: Env, logProb: Double,
    context: PnpInferenceContext, tag: Any, choice: Any): Unit = {
    val stateLogProb = context.computeScore(tag, choice, env) + logProb
    if (stateLogProb > Double.NegativeInfinity) {
      queue.offer(SearchState(value, env, stateLogProb, tag, choice), stateLogProb)
    }
  }
}

class EnumeratePnpSearchQueue[A] (
    val finished: PnpSearchQueue[A]
) extends PnpSearchQueue[A] {
  val endContinuation = new PnpEndContinuation[A]
  
  override def offer(value: Pnp[A], env: Env, logProb: Double,
      context: PnpInferenceContext, tag: Any, choice: Any): Unit = {
    val stateLogProb = context.computeScore(tag, choice, env) + logProb
    if (stateLogProb > Double.NegativeInfinity) {
      value.searchStep(env, stateLogProb, context, endContinuation, this, finished)
    }
  }
}

class ContinuationPnpSearchQueue[A, B] (
    val queue: PnpSearchQueue[B],
    val cont: PnpContinuation[A,B]
) extends PnpSearchQueue[A] {
  
  override def offer(value: Pnp[A], env: Env, logProb: Double, context: PnpInferenceContext,
      tag: Any, choice: Any): Unit = {
    queue.offer(BindPnp(value, cont), env, logProb, context, tag, choice)
  }
}

case class SearchState[A](val value: Pnp[A], val env: Env, val logProb: Double, val tag: Any, val choice: Any) {
}
