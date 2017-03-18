package org.allenai.pnp

sealed trait PnpContinuation[A, B] {
  def prepend[D](g: D => Pnp[A]): PnpContinuation[D, B]

  def append[D](g: B => Pnp[D]): PnpContinuation[A, D] = {
    append(PnpContinuationFunction(g))
  }
  def append[D](g: PnpContinuation[B, D]): PnpContinuation[A, D]

  def searchStep(arg: A, env: Env, inferenceState: PnpInferenceState, queue: PnpSearchQueue[B],
      finished: PnpSearchQueue[B]): Unit
      
  def sampleStep(arg: A, env: Env, inferenceState: PnpInferenceState,  queue: PnpSearchQueue[B],
      finished: PnpSearchQueue[B]): Unit
}

case class PnpContinuationFunction[A, B](val f: A => Pnp[B]) extends PnpContinuation[A, B] {
  val endContinuation = new PnpEndContinuation[B]
  
  override def prepend[D](g: D => Pnp[A]): PnpContinuation[D, B] = {
    PnpContinuationChain(g, this)
  }
  
  override def append[D](g: PnpContinuation[B, D]): PnpContinuation[A, D] = {
    PnpContinuationChain(f, g)
  }
  
  override def searchStep(arg: A, env: Env, inferenceState: PnpInferenceState, queue: PnpSearchQueue[B],
      finished: PnpSearchQueue[B]): Unit = {
    f(arg).searchStep(env, inferenceState, endContinuation, queue, finished)
  }
  
  override def sampleStep(arg: A, env: Env, inferenceState: PnpInferenceState, queue: PnpSearchQueue[B],
      finished: PnpSearchQueue[B]): Unit = {
    f(arg).sampleStep(env, inferenceState, endContinuation, queue, finished)
  }
}

case class PnpContinuationChain[A, B, C](val f: A => Pnp[B], val cont: PnpContinuation[B, C])
  extends PnpContinuation[A, C] {
  
  override def prepend[D](g: D => Pnp[A]): PnpContinuation[D, C] = {
    PnpContinuationChain(g, this)
  }
  
  override def append[D](g: PnpContinuation[C, D]): PnpContinuation[A, D] = {
    PnpContinuationChain(f, cont.append(g))
  }
  
  override def searchStep(arg: A, env: Env, inferenceState: PnpInferenceState, queue: PnpSearchQueue[C],
      finished: PnpSearchQueue[C]): Unit = {
    f(arg).searchStep(env, inferenceState, cont, queue, finished)
  }
  
  override def sampleStep(arg: A, env: Env, inferenceState: PnpInferenceState, queue: PnpSearchQueue[C],
      finished: PnpSearchQueue[C]): Unit = {
    f(arg).sampleStep(env, inferenceState, cont, queue, finished)
  }
}

case class PnpEndContinuation[A]() extends PnpContinuation[A, A] {
  override def prepend[D](g: D => Pnp[A]): PnpContinuation[D, A] = {
    PnpContinuationChain(g, this)
  }

  override def append[D](g: PnpContinuation[A, D]): PnpContinuation[A, D] = {
    if (g.isInstanceOf[PnpEndContinuation[A]]) {
      return this.asInstanceOf[PnpContinuation[A,D]]
    } else {
      throw new UnsupportedOperationException("Cannot append to the end continuation")
    }
  }

  override def searchStep(arg: A, env: Env, inferenceState: PnpInferenceState, queue: PnpSearchQueue[A],
      finished: PnpSearchQueue[A]): Unit = {
    finished.offer(Pnp.value(arg), env, inferenceState, null, null, env)
  }

  override def sampleStep(arg: A, env: Env, inferenceState: PnpInferenceState, queue: PnpSearchQueue[A],
      finished: PnpSearchQueue[A]): Unit = {
    finished.offer(Pnp.value(arg), env, inferenceState, null, null, env)
  }
}