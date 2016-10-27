package org.allenai.pnp

import com.jayantkrish.jklol.inference.MarginalCalculator.ZeroProbabilityError
import com.jayantkrish.jklol.models.parametric.SufficientStatistics
import com.jayantkrish.jklol.training.GradientOracle
import com.jayantkrish.jklol.training.LogFunction

/** Loglikelihood training oracle for neural probabilistic programs.
  * Each training example consists of two distributions: a
  * conditional distribution C(e) over correct program executions e and
  * an unconditional distribution U(e) over all program executions. This
  * class implements the objective log (sum_e C(e) / sum_e U(e)),
  * which is the loglikelihood of correct executions.
  */
class PpLoglikelihoodOracle[A](beamSize: Int, family: ParametricPpModel)
    extends GradientOracle[PpModel, PpExample[A]] {

  override def initializeGradient(): SufficientStatistics = {
    family.getNewSufficientStatistics
  }

  override def instantiateModel(params: SufficientStatistics): PpModel = {
    family.getModelFromParameters(params)
  }

  override def accumulateGradient(
    gradient: SufficientStatistics,
    currentParameters: SufficientStatistics, model: PpModel,
    example: PpExample[A], log: LogFunction
  ): Double = {
    val env = example.env
    val graph = model.getInitialComputationGraph

    // Compute the distribution over correct executions.
    log.startTimer("pp_loglikelihood/conditional")
    val conditional = example.conditional.beamSearchWithFilter(beamSize, env,
      example.conditionalFilter, graph, log)
    val conditionalPartitionFunction = conditional.partitionFunction
    log.stopTimer("pp_loglikelihood/conditional")

    // Throw a search error if we cannot find at least one
    // correct execution.
    if (conditional.executions.size == 0 || conditionalPartitionFunction == 0.0) {
      println("conditional search failure")
      throw new ZeroProbabilityError()
    }

    // Compute the unconditional distribution over 
    // all executions.
    log.startTimer("pp_loglikelihood/unconditional")
    val unconditional = example.unconditional.beamSearchWithFilter(beamSize, env,
      (x: Env) => true, graph, log)
    val unconditionalPartitionFunction = unconditional.partitionFunction
    log.stopTimer("pp_loglikelihood/unconditional")

    if (unconditional.executions.size == 0 || unconditionalPartitionFunction == 0.0) {
      println("unconditional search failure")
      for (ex <- unconditional.executions) {
        println(ex.logProb + " " + ex.prob + " " + ex.value)
      }
      throw new ZeroProbabilityError()
    }

    // Increment gradient. The gradient is the difference of 
    // feature expectations between the conditional and 
    // unconditional distributions.
    log.startTimer("pp_loglikelihood/increment_gradient")
    family.incrementSufficientStatistics(gradient, conditional, 1.0 / conditional.partitionFunction)
    family.incrementSufficientStatistics(gradient, unconditional,
      -1.0 / unconditional.partitionFunction)
    log.stopTimer("pp_loglikelihood/increment_gradient")

    // This is an approximation of the loglikelihood because
    // we're running approximate inference.
    Math.log(conditionalPartitionFunction) - Math.log(unconditionalPartitionFunction)
  }
}