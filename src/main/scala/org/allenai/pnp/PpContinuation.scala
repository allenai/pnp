package org.allenai.pnp

sealed trait PpContinuation[A, B] {
  def prepend[D](g: D => Pp[A]): PpContinuation[D, B]

  def append[D](g: B => Pp[D]): PpContinuation[A, D] = {
    append(PpContinuationFunction(g))
  }
  def append[D](g: PpContinuation[B, D]): PpContinuation[A, D]

  def searchStep(arg: A, env: Env, logProb: Double, queue: PpSearchQueue[B],
      finished: PpSearchQueue[B]): Unit 
}

case class PpContinuationFunction[A, B](val f: A => Pp[B]) extends PpContinuation[A, B] {
  override def prepend[D](g: D => Pp[A]): PpContinuation[D, B] = {
    PpContinuationChain(g, this)
  }
  
  override def append[D](g: PpContinuation[B, D]): PpContinuation[A, D] = {
    PpContinuationChain(f, g)
  }
  
  override def searchStep(arg: A, env: Env, logProb: Double, queue: PpSearchQueue[B],
      finished: PpSearchQueue[B]): Unit = {
    f(arg).lastSearchStep(env, logProb, queue, finished)
  }
}

case class PpContinuationChain[A, B, C](val f: A => Pp[B], val cont: PpContinuation[B, C])
  extends PpContinuation[A, C] {
  
  override def prepend[D](g: D => Pp[A]): PpContinuation[D, C] = {
    PpContinuationChain(g, this)
  }
  
  override def append[D](g: PpContinuation[C, D]): PpContinuation[A, D] = {
    PpContinuationChain(f, cont.append(g))
  }
  
  override def searchStep(arg: A, env: Env, logProb: Double, queue: PpSearchQueue[C],
      finished: PpSearchQueue[C]): Unit = {
    f(arg).searchStep(env, logProb, cont, queue, finished)
  }
}

/*
case class PpEndContinuation[A](val queue: PpSearchQueue[B])
  extends PpContinuation[A, A] {
  
  override def prepend[D](g: D => Pp[A]): PpContinuation[D, A] = {
    PpContinuationChain(g, this)
  }
  
  override def append[D](g: PpContinuation[B, D]): PpContinuation[A, D] = {
    throw new UnsupportedOperationException()
  }
  
  override def apply(arg: A): Pp[C] = {
    return EndPp(arg, queue)
  }
}
*/
