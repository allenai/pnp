package org.allenai.p3

import com.jayantkrish.jklol.inference.MarginalCalculator.ZeroProbabilityError
import com.jayantkrish.jklol.models.parametric.SufficientStatistics
import com.jayantkrish.jklol.training.GradientOracle
import com.jayantkrish.jklol.training.LogFunction

/** Loglikelihood training objective for P3. Each training
  * example consists of a sentence and a set of correct
  * CCG parses/logical forms/executions (represented using
  * filters). This objective trains the parameters of a
  * CCG parser and neural probabilistic program to maximize
  * the probability of predicting correct
  * parses/logical forms/executions.
  */
class P3PpLoglikelihoodOracle(
    val family: ParametricP3PpModel,
    val inference: P3PpBeamInference
) extends GradientOracle[P3PpModel, P3PpExample] {

  override def initializeGradient(): SufficientStatistics = {
    family.getNewSufficientStatistics()
  }

  override def instantiateModel(parameters: SufficientStatistics): P3PpModel = {
    return family.getModelFromParameters(parameters)
  }

  override def accumulateGradient(
    gradient: SufficientStatistics,
    currentParameters: SufficientStatistics, model: P3PpModel,
    example: P3PpExample, log: LogFunction
  ): Double = {
    val sentence = example.sentence
    val env = example.env

    println(sentence)
    println(env)

    // Get a distribution on executions conditioned on the label of the example.
    // Do this first because it's faster, so search errors take less time to process.
    log.startTimer("update_gradient/output_marginal")
    val conditionalParsesInit = inference.beamSearch(
      model, sentence, env, example.chartCost, example.executionCost, log
    )

    log.startTimer("update_gradient/output_marginal/filter")
    val conditionalParses = conditionalParsesInit.conditionLfs(example.lfFilter)
      .conditionExecutions(example.executionFilter)
    log.stopTimer("update_gradient/output_marginal/filter")

    if (conditionalParses.partitionFunction == 0.0) {
      println("Search error (Correct): " + sentence)
      throw new ZeroProbabilityError()
    }
    log.stopTimer("update_gradient/output_marginal")

    // Get a distribution over unconditional executions.
    log.startTimer("update_gradient/input_marginal")

    // System.out.println("unconditional evaluations:")
    val unconditionalParses = if (example.chartCost == null && example.executionCost == null) {
      conditionalParsesInit
    } else {
      inference.beamSearch(model, sentence, env, log)
    }

    if (unconditionalParses.partitionFunction == 0.0) {
      System.out.println("Search error (Predicted): " + sentence)
      throw new ZeroProbabilityError()
    }
    log.stopTimer("update_gradient/input_marginal")

    /*
    println("conditional: ")
    for (p <- conditionalParses.parses) {
      println(p)
    }
    println(conditionalParses.partitionFunction)
    
    println("unconditional: ")
    for (p <- unconditionalParses.parses) {
      println(p)
    }
    println(unconditionalParses.partitionFunction)
    */

    log.startTimer("update_gradient/increment_gradient")
    family.incrementSufficientStatistics(gradient, currentParameters,
      sentence, unconditionalParses, -1.0 / unconditionalParses.partitionFunction, log)

    family.incrementSufficientStatistics(gradient, currentParameters,
      sentence, conditionalParses, 1.0 / conditionalParses.partitionFunction, log)
    log.stopTimer("update_gradient/increment_gradient")

    // Note that the returned loglikelihood is an approximation because
    // inference is approximate.
    Math.log(conditionalParses.partitionFunction) - Math.log(unconditionalParses.partitionFunction)
  }
}