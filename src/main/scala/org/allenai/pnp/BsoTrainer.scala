package org.allenai.pnp

import scala.collection.mutable.ListBuffer

import com.google.common.base.Preconditions
import com.jayantkrish.jklol.training.LogFunction
import com.jayantkrish.jklol.util.KbestQueue

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._
import scala.util.Random

/**
 * Beam search trainer implementing a LaSO-like algorithm.
 * This trainer implements a margin loss between the
 * lowest-scoring item on the beam and the highest-scoring
 * correct execution. Thus, a gradient update is performed
 * every time all correct executions fall off the beam.
 */
class BsoTrainer(val epochs: Int, val beamSize: Int, val maxIters: Int,
    val model: PnpModel, val trainer: Trainer, val log: LogFunction) {

  Preconditions.checkArgument(model.locallyNormalized == false,
      "BsoTrainer expects model to be not locally normalized".asInstanceOf[Any])

  import DyNetScalaHelpers._

  def train[A](examples: Seq[PnpExample[A]]): Unit = {
    for (i <- 0 until epochs) {
      var loss = 0.0
      log.notifyIterationStart(i)
      for (example <- Random.shuffle(examples)) {
        loss +=  doExampleUpdate(example)
      }
      
      trainer.update_epoch()

      log.logStatistic(i, "loss", loss)
      log.notifyIterationEnd(i)
    }
  }

  private def doExampleUpdate[A](example: PnpExample[A]): Double = {
    val cg = ComputationGraph.getNew
    var loss = 0.0

    val env = example.env
    val stateCost = if (example.conditionalExecutionScore == null) {
      ExecutionScore.zero
    } else {
      example.conditionalExecutionScore
    }
    val graph = model.getComputationGraph(cg)

    val queue = new BsoPnpQueue[A](beamSize, stateCost, graph, log)
    val finished = new BsoPnpQueue[A](beamSize, stateCost, graph, log)
    val endContinuation = new PnpEndContinuation[A]()

    val startEnv = env.setLog(log)
    queue.offer(example.unconditional, env, 0.0, null, null, env)

    log.startTimer("bso/beam_search")
    val beam = new Array[SearchState[A]](beamSize)
    var numIters = 0
    val losses = ListBuffer[Expression]()
    while (queue.queue.size > 0 &&
        (maxIters < 0 || numIters < maxIters)) {
      // println(numIters + " " + queue.queue.size)

      // TODO: check this
      val beamSize = queue.queue.size
      
      // Check margin constraint.
      // The highest-scoring correct execution's score must exceed
      // that of the lowest-scoring execution on beam by a margin of 1.

      // Note that beam(0) is the lowest-scoring element   
      // val (beamEx, beamCost) = queue.queue.getItems()(0)
      log.startTimer("bso/beam_search/get_executions")
      val worstIncorrectEx = if (queue.incorrectQueue.size > 0) {
        queue.incorrectQueue.getItems()(0) 
      } else {
        null
      }

      val bestCorrectEx = if (queue.correctQueue.size > 0) {
        queue.correctQueue.getItems.slice(
            0, queue.correctQueue.size).maxBy(x => x.logProb)
      } else {
        null
      }
      log.stopTimer("bso/beam_search/get_executions")

      var nextBeamSize = -1
      if (numIters != 0 && bestCorrectEx != null && worstIncorrectEx != null &&
          worstIncorrectEx.logProb + 1 > bestCorrectEx.logProb) {
        // Margin violation
        // println("m: " + numIters + " " + worstIncorrectEx.logProb + " " + bestCorrectEx.logProb)

        // Add to the loss.
        log.startTimer("bso/beam_search/margin_violation")
        
        val beamScoreExpr = worstIncorrectEx.env.getScore(false, cg)
        val correctScoreExpr = bestCorrectEx.env.getScore(false, cg)

        // XXX: allow different costs.
        losses += ((beamScoreExpr + 1) - correctScoreExpr)

        // Continue the search with the best correct execution.
        beam(0) = bestCorrectEx
        nextBeamSize = 1
        
        log.stopTimer("bso/beam_search/margin_violation")
      } else {
        // No margin violation. Queue up all beam executions for 
        // the next search step.
        for (i <- 0 until beamSize) {
          beam(i) = queue.queue.getItems()(i)._1 
        }

        nextBeamSize = beamSize
      }

      queue.queue.clear
      queue.correctQueue.clear
      queue.incorrectQueue.clear

      // Continue beam search.
      log.startTimer("bso/beam_search/search_step")
      for (i <- 0 until nextBeamSize) {
        val state = beam(i)
        state.value.searchStep(state.env, state.logProb, endContinuation, queue, finished)
      }
      log.stopTimer("bso/beam_search/search_step")

      numIters += 1
    }
    log.stopTimer("bso/beam_search")
              
    // Compute margin loss for final highest-scoring incorrect entry
    // vs. correct.
    val finalBestIncorrect = if (finished.incorrectQueue.size > 0) {
      finished.incorrectQueue.getItems.slice(
          0, finished.incorrectQueue.size).maxBy(_.logProb)
    } else {
      null
    }
        
    val finalBestCorrect = if (finished.correctQueue.size > 0) {
      finished.correctQueue.getItems.slice(
          0, finished.correctQueue.size).maxBy(_.logProb)
    } else {
      null
    }

    if (finalBestIncorrect != null && finalBestCorrect != null &&
        finalBestIncorrect.logProb + 1 > finalBestCorrect.logProb) {
        // Margin violation
      // println("m: end " + finalBestIncorrect.logProb + " " + finalBestCorrect.logProb)

      // Add to the loss.
      val beamScoreExpr = finalBestIncorrect.env.getScore(false, cg)
      val correctScoreExpr = finalBestCorrect.env.getScore(false, cg)

      // XXX: allow different costs.
      losses += ((beamScoreExpr + 1) - correctScoreExpr)
    }

    if (losses.size > 0) {
      val lossExpr = sum(new ExpressionVector(losses))
      
      log.startTimer("bso/eval_loss")
      loss += as_scalar(cg.incremental_forward(lossExpr)) 
      log.stopTimer("bso/eval_loss")
      
      log.startTimer("bso/backward")
      cg.backward(lossExpr)
      trainer.update(1.0f)
      log.stopTimer("bso/backward")
    }

    loss
  }
}

class BsoPnpQueue[A](size: Int, val stateCost: ExecutionScore,
    val graph: CompGraph, val log: LogFunction) extends PnpSearchQueue[A] {

  val queue = new KbestQueue(size, Array.empty[(SearchState[A], Double)])
  val correctQueue = new KbestQueue(size, Array.empty[SearchState[A]])
  val incorrectQueue = new KbestQueue(size, Array.empty[SearchState[A]])
  
  val EXECUTION_INCORRECT_VAR_NAME = "**bso_execution_incorrect**"

  override def offer(value: Pnp[A], env: Env, logProb: Double, tag: Any,
      choice: Any, myEnv: Env): Unit = {
    if (logProb > Double.NegativeInfinity) {
      log.startTimer("bso/beam_search/search_step/eval_cost")
      val cost = stateCost(tag, choice, env)
      log.stopTimer("bso/beam_search/search_step/eval_cost")

      val nextEnv = if (cost != 0.0) {
        env.setVar(EXECUTION_INCORRECT_VAR_NAME, null)
      } else {
        env
      }
      
      val state = SearchState(value, nextEnv, logProb, tag, choice)
      queue.offer((state, cost), logProb)

      if (nextEnv.isVarBound(EXECUTION_INCORRECT_VAR_NAME)) {
        incorrectQueue.offer(state, logProb)
      } else {
        correctQueue.offer(state, logProb)
      }
    }
  }
}
