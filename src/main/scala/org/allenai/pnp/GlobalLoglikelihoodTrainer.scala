package org.allenai.pnp

import com.google.common.base.Preconditions
import com.jayantkrish.jklol.training.LogFunction

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._

class GlobalLoglikelihoodTrainer(val epochs: Int, val beamSize: Int,
    val maxSearchSteps: Int, val model: PpModel, val trainer: Trainer,
    val logFn: LogFunction) {

  import DynetScalaHelpers._
  
  def train[A](examples: List[PpExample[A]]): Unit = {
    for (i <- 0 until epochs) {
      var loss = 0.0
      var searchErrors = 0
      logFn.notifyIterationStart(i)
      for (example <- examples) {
        val cg = new ComputationGraph
       
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

        cg.delete()
      }
      logFn.logStatistic(i, "search errors", searchErrors) 
      // println(i + "  loss: " + loss)
      trainer.update_epoch()
    }
  }
  
  private def marginalsToLogProbExpression[A](marginals: PpBeamMarginals[A]): Expression = {
    val exScores = for {
      ex <- marginals.executions
    } yield {
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
      exScore
    }
    
    if (exScores.length == 0) {
      null 
    } else if (exScores.length == 1) {
      exScores(0)
    } else {
      logsumexp(new ExpressionVector(exScores))
    }
  }
}