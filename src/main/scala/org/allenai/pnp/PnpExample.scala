package org.allenai.pnp

import ExecutionScore.ExecutionScore

/** A training example for neural probabilistic programs. An example
  * consists of a conditional and an unconditional program, and an
  * environment in which these programs execute. An additional
  * filter on environments may be provided to further restrict the set
  * of conditional executions during inference.
  */
case class PnpExample[A](unconditional: Pnp[A], conditional: Pnp[A],
    env: Env, conditionalExecutionScore: ExecutionScore) {
}

object PnpExample {
  def fromDistributions[A](unconditional: Pnp[A], conditional: Pnp[A]) = {
    PnpExample[A](unconditional, conditional, Env.init, ExecutionScore.zero)
  }
}