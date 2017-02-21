package org.allenai.pnp

import com.jayantkrish.jklol.training.LogFunction
import com.jayantkrish.jklol.util.KbestQueue

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._

import scala.collection.mutable.ListBuffer
import com.google.common.base.Preconditions

class BsoTrainer(val epochs: Int, val beamSize: Int, val maxIters: Int,
    val model: PpModel, val trainer: Trainer, val log: LogFunction) {

  Preconditions.checkArgument(model.locallyNormalized == false)

  import DynetScalaHelpers._

  def train[A](examples: Seq[PpExample[A]]): Unit = {
    for (i <- 0 until epochs) {
      var loss = 0.0
      var searchErrors = 0.0
      log.notifyIterationStart(i)
      for (example <- examples) {
        val (l, s) = doExampleUpdate(example)
        
        if (s) {
          loss += l
        } else {
          searchErrors += 1
        }
      }
      
      trainer.update_epoch()

      log.logStatistic(i, "loss", loss)
      log.logStatistic(i, "search errors", searchErrors)
      log.notifyIterationEnd(i)
    }
  }
  
  private def doExampleUpdate[A](example: PpExample[A]): (Double, Boolean) = {
    val cg = new ComputationGraph
    var loss = 0.0

    val env = example.env
    val stateCost = if (example.conditionalExecutionScore == null) {
      ExecutionScore.zero
    } else {
      example.conditionalExecutionScore
    }
    val graph = model.getComputationGraph(cg)

    val queue = new BsoPpQueue[A](beamSize, stateCost, graph, log)
    val finished = new BsoPpQueue[A](beamSize, stateCost, graph, log)

    val startEnv = env.setLog(log)
    queue.offer(example.unconditional, env, 0.0, null, null, env)

    val beam = new Array[SearchState[A]](beamSize)
    var numIters = 0
    val losses = ListBuffer[Expression]()
    while (queue.queue.size > 0 && queue.correctQueue.size > 0 &&
        (maxIters < 0 || numIters < maxIters)) {
      numIters += 1
      // println(numIters + " " + queue.queue.size)

      // TODO: check this
      val beamSize = queue.queue.size
      
      // Check margin constraint.
      // The highest-scoring correct execution's score must exceed
      // that of the lowest-scoring execution on beam by a margin of 1.

      // Note that beam(0) is the lowest-scoring element   
      val (beamEx, beamCost) = queue.queue.getItems()(0)
      val bestCorrectEx = queue.correctQueue.getItems.slice(
          0, queue.correctQueue.size).maxBy(x => x.logProb)
      
      var nextBeamSize = -1
      if (beamEx.logProb > bestCorrectEx.logProb + 1) {
        // Margin violation
        
        // Add to the loss.
        val beamScoreExpr = sum(new ExpressionVector(
            beamEx.env.labelNodeIds.zip(beamEx.env.labels).map(x => pick(x._1, x._2))))
        val correctScoreExpr = sum(new ExpressionVector(
            bestCorrectEx.env.labelNodeIds.zip(bestCorrectEx.env.labels).map(x => pick(x._1, x._2))))
            
        // XXX: allow different costs.
        losses += (beamScoreExpr - correctScoreExpr)

        // Continue the search with the best correct execution.
        beam(0) = bestCorrectEx
        nextBeamSize = 1
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

      // Continue beam search.
      for (i <- 0 until nextBeamSize) {
        val state = beam(i)
        state.value.lastSearchStep(state.env, state.logProb, queue, finished)
      }
    }
    
    // TODO: loss for final beam top vs. correct.

    if (losses.size > 0) {
      val lossExpr = sum(new ExpressionVector(losses))
      
      log.startTimer("pp_bso/eval_loss")
      loss += as_scalar(cg.incremental_forward(lossExpr)) 
      log.stopTimer("pp_bso/eval_loss")
      
      log.startTimer("pp_bso/backward")
      cg.backward(lossExpr)
      trainer.update(1.0f)
      log.stopTimer("pp_bso/backward")
    }

    cg.delete()

    (loss, true)
  }
}

class BsoPpQueue[A](size: Int, val stateCost: ExecutionScore,
    val graph: CompGraph, val log: LogFunction) extends PpSearchQueue[A] {

  val queue = new KbestQueue(size, Array.empty[(SearchState[A], Double)])
  val correctQueue = new KbestQueue(size, Array.empty[SearchState[A]])

  override def offer(value: Pp[A], env: Env, logProb: Double, tag: Any,
      choice: Any, myEnv: Env): Unit = {
    if (logProb > Double.NegativeInfinity) {
      val cost = stateCost(tag, choice, env)

      val state = SearchState(value, env, logProb, tag, choice)
      queue.offer((state, cost), logProb)

      if (cost == 0.0) {
        correctQueue.offer(state, logProb)
      }
    }
  }
}
