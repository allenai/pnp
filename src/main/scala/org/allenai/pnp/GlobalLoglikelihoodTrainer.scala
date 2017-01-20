package org.allenai.pnp

import org.allenai.pnp.examples.DynetScalaHelpers

import com.google.common.base.Preconditions
import com.jayantkrish.jklol.training.LogFunction

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._

class GlobalLoglikelihoodTrainer(val epochs: Int, val beamSize: Int,
    val model: PpModel, val trainer: Trainer, val logFn: LogFunction) {

  import DynetScalaHelpers._
  
  def train[A](examples: List[PpExample[A]]): Unit = {
    for (i <- 0 until epochs) {
      var loss = 0.0
      for (example <- examples) {
        val cg = new ComputationGraph
       
        val env = example.env
        val graph = model.getInitialComputationGraph(cg)
       
        // Compute the distribution over correct executions.
        logFn.startTimer("pp_loglikelihood/conditional")
        val conditional = example.conditional.beamSearch(beamSize, -1, env,
            example.conditionalExecutionScore, graph, logFn)
        val conditionalPartitionFunction = conditional.partitionFunction
        logFn.stopTimer("pp_loglikelihood/conditional")
        
        // TODO: handle search errors
        
        // Compute the unconditional distribution over 
        // all executions.
        logFn.startTimer("pp_loglikelihood/unconditional")
        val unconditional = example.unconditional.beamSearchWithFilter(beamSize, env,
            (x: Env) => true, graph, logFn)
        val unconditionalPartitionFunction = unconditional.partitionFunction
        logFn.stopTimer("pp_loglikelihood/unconditional")

        val conditionalLogSumProb = marginalsToLogProbExpression(conditional)
        val unconditionalLogSumProb = marginalsToLogProbExpression(unconditional)
        val lossExpr = unconditionalLogSumProb - conditionalLogSumProb

        loss += as_scalar(cg.incremental_forward(lossExpr))
        cg.backward(lossExpr)
        trainer.update(1.0f)
        cg.delete()
      }
      // println(i + "  loss: " + loss)
      trainer.update_epoch()
    }
  }
  
  private def marginalsToLogProbExpression[A](marginals: PpBeamMarginals[A]): Expression = {
    var lossExpr: Expression = null
    for (ex <- marginals.executions) {
      val labeledExpressions = ex.env.labelNodeIds
      val labelIndexes = ex.env.labels
      var exScore: Expression = null
      for ((expr, labelInd) <- labeledExpressions.zip(labelIndexes)) {
        val decisionScore = pick(expr, labelInd)
        if (exScore == null) {
          exScore = decisionScore
        } else {
          exScore = exScore + decisionScore
        }
      }

      if (lossExpr == null) {
        lossExpr = exp(exScore)
      } else {
        lossExpr = lossExpr + exp(exScore)
      }
    }

    log(lossExpr)
  }
}