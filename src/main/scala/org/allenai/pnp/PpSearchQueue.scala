package org.allenai.pnp

import com.jayantkrish.jklol.training.LogFunction
import com.jayantkrish.jklol.util.KbestQueue

sealed trait PpSearchQueue[A] {
  val graph: CompGraph
  val stateCost: ExecutionScore
  val log: LogFunction

  def offer(value: Pp[A], env: Env, logProb: Double, tag: Any, choice: Any, myEnv: Env): Unit
}

class BeamPpSearchQueue[A](size: Int, val stateCost: ExecutionScore,
    val graph: CompGraph, val log: LogFunction) extends PpSearchQueue[A] {

  /*
  val supplier = new Supplier[SearchState2]() {
    def get: SearchState2 = {
      new SearchState2(null, null, 0.0, null)
    }
  }
  val pool = new ObjectPool(supplier, size + 1, Array.empty[SearchState2])
  */
  
  val queue = new KbestQueue(size, Array.empty[SearchState[A]])

  override def offer(value: Pp[A], env: Env, logProb: Double, tag: Any,
      choice: Any, myEnv: Env): Unit = {
    val stateLogProb = stateCost(tag, choice, env) + logProb
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

case class SearchState[A](value: Pp[A], env: Env, logProb: Double, tag: Any, choice: Any) {
}
