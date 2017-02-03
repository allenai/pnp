package org.allenai.pnp

trait ExecutionScore {
  def apply(tag: Any, choice: Any, env: Env): Double
}

object ExecutionScore {
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