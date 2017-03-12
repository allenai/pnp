package org.allenai.pnp

trait ExecutionScore {
  def apply(tag: Any, choice: Any, env: Env): Double
}

object ExecutionScore {
  // TODO: make this a type so we can compose it with other functions, etc.
  // tag, choice, env
  // type ExecutionScore = (Any, Any, Env) => Double

  val zero = new ExecutionScore() {
    def apply(tag: Any, choice: Any, env: Env): Double = {
        0.0
    }
  }
  
  def fromFilter(keepState: Env => Boolean): ExecutionScore = {
    new ExecutionScore() {
      def apply(tag: Any, choice: Any, env: Env): Double = {
        if (keepState(env)) {
          0.0
        } else {
          Double.NegativeInfinity
        }
      }
    }
  }
}