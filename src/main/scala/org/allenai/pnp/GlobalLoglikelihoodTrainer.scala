package org.allenai.pnp

import com.google.common.base.Preconditions
import com.jayantkrish.jklol.training.LogFunction

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._

class GlobalLoglikelihoodTrainer(val epochs: Int, val beamSize: Int,
    val maxSearchSteps: Int, val model: PnpModel, val trainer: Trainer,
    val logFn: LogFunction) {

  import DynetScalaHelpers._
  
  def train[A](examples: Seq[PnpExample[A]]): Unit = {
    for (i <- 0 until epochs) {
      var loss = 0.0
      var searchErrors = 0
      logFn.notifyIterationStart(i)
      for (example <- examples) {
        val cg = ComputationGraph.getNew
       
        val env = example.env
        val graph = model.getComputationGraph(cg)
       
        // Compute the distribution over correct executions.
        logFn.startTimer("pp_loglikelihood/conditional")
        val conditional = example.conditional.beamSearch(beamSize, maxSearchSteps, env,
            example.conditionalExecutionScore, graph, logFn)
        val conditionalPartitionFunction = conditional.partitionFunction
        logFn.stopTimer("pp_loglikelihood/conditional")
        
        // TODO: handle search errors
        
        // Compute the unconditional distribution over 
        // all executions.
        logFn.startTimer("pp_loglikelihood/unconditional")
        val unconditional = example.unconditional.beamSearch(beamSize, maxSearchSteps,
            env, null, graph, logFn)
        val unconditionalPartitionFunction = unconditional.partitionFunction
        logFn.stopTimer("pp_loglikelihood/unconditional")

        val conditionalLogSumProb = marginalsToLogProbExpression(conditional)
        val unconditionalLogSumProb = marginalsToLogProbExpression(unconditional)
        
        if (conditionalLogSumProb != null && unconditionalLogSumProb != null) {
          val lossExpr = unconditionalLogSumProb - conditionalLogSumProb

          loss += as_scalar(cg.incremental_forward(lossExpr))
          cg.backward(lossExpr)
          trainer.update(1.0f)
        } else {
          searchErrors += 1
        }
      }
      logFn.logStatistic(i, "search errors", searchErrors) 
      // println(i + "  loss: " + loss)
      trainer.update_epoch()
    }
  }
  
  private def marginalsToLogProbExpression[A](marginals: PnpBeamMarginals[A]): Expression = {
    val exScores = marginals.executions.map(_.env.getScore(false))
    
    if (exScores.length == 0) {
      null 
    } else if (exScores.length == 1) {
      exScores(0)
    } else {
      logsumexp(new ExpressionVector(exScores))
    }
  }
}