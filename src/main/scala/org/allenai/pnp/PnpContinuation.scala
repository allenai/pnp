package org.allenai.pnp

sealed trait PnpContinuation[A, B] {
  def prepend[D](g: D => Pnp[A]): PnpContinuation[D, B]

  def append[D](g: B => Pnp[D]): PnpContinuation[A, D] = {
    append(PnpContinuationFunction(g))
  }
  def append[D](g: PnpContinuation[B, D]): PnpContinuation[A, D]

  def searchStep(arg: A, env: Env, logProb: Double, queue: PnpSearchQueue[B],
      finished: PnpSearchQueue[B]): Unit 
}

case class PnpContinuationFunction[A, B](val f: A => Pnp[B]) extends PnpContinuation[A, B] {
  override def prepend[D](g: D => Pnp[A]): PnpContinuation[D, B] = {
    PnpContinuationChain(g, this)
  }
  
  override def append[D](g: PnpContinuation[B, D]): PnpContinuation[A, D] = {
    PnpContinuationChain(f, g)
  }
  
  override def searchStep(arg: A, env: Env, logProb: Double, queue: PnpSearchQueue[B],
      finished: PnpSearchQueue[B]): Unit = {
    f(arg).lastSearchStep(env, logProb, queue, finished)
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
  
  override def searchStep(arg: A, env: Env, logProb: Double, queue: PnpSearchQueue[C],
      finished: PnpSearchQueue[C]): Unit = {
    f(arg).searchStep(env, logProb, cont, queue, finished)
  }
}
