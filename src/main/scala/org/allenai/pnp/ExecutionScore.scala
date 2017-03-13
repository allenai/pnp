package org.allenai.pnp

object ExecutionScore {
  
  /**
   * ExecutionScore is a function from a tag
   * (i.e., a name for a choice point), a choice,
   * and an env to a score for the choice.
   */
  type ExecutionScore = (Any, Any, Env) => Double

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