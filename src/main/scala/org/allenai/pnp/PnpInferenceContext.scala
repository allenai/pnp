package org.allenai.pnp

import com.jayantkrish.jklol.training.{LogFunction, NullLogFunction}

class PnpInferenceContext(
  cg: CompGraph = null,
  val log: LogFunction = new NullLogFunction(),
  activeScores: Set[ExecutionScore] = Set.empty) {

  def compGraph: CompGraph = {
    assert (cg != null)
    cg
  }

  def addExecutionScore(es: ExecutionScore) = new PnpInferenceContext(cg, log, activeScores + es)
  def removeExecutionScore(es: ExecutionScore) = new PnpInferenceContext(cg, log, activeScores - es)

  def computeScore(tag: Any, choice: Any, env: Env): Double =
    activeScores.map(_(tag, choice, env)).sum


  def setLog(newLog: LogFunction): PnpInferenceContext = {
    new PnpInferenceContext(cg, newLog, activeScores)
  }
}

object PnpInferenceContext {
  def init: PnpInferenceContext = new PnpInferenceContext()
  def init(cg: CompGraph): PnpInferenceContext = new PnpInferenceContext(cg)
  def init(model: PnpModel): PnpInferenceContext = init(model.getComputationGraph())
}