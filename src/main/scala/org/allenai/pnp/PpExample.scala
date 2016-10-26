package org.allenai.pnp

/** A training example for neural probabilistic programs. An example
  * consists of a conditional and an unconditional program, and an
  * environment in which these programs execute. An additional
  * filter on environments may be provided to further restrict the set
  * of conditional executions during inference.
  */
case class PpExample[A](unconditional: Pp[A], conditional: Pp[A],
    env: Env, conditionalFilter: Env => Boolean) {
}

object PpExample {
  def fromDistributions[A](unconditional: Pp[A], conditional: Pp[A]) = {
    PpExample[A](unconditional, conditional, Env.init, x => true)
  }
}