package org.allenai.pnp

class PnpInferenceState(
  cg: CompGraph = null,
  val logProb: Double = 0.0,
  activeScores: Set[ExecutionScore] = Set.empty) {

  def compGraph: CompGraph = {
    assert (cg != null)
    cg
  }

  def addExecutionScore(es: ExecutionScore) = new PnpInferenceState(cg, logProb, activeScores + es)
  def removeExecutionScore(es: ExecutionScore) = new PnpInferenceState(cg, logProb, activeScores - es)

  def withLogProb(newLogProb: Double) = new PnpInferenceState(cg, newLogProb, activeScores)
  def computeScore(tag: Any, choice: Any, env: Env): Double =
    activeScores.map(_(tag, choice, env)).sum
}

object PnpInferenceState {
  def init: PnpInferenceState = new PnpInferenceState()
  def init(cg: CompGraph): PnpInferenceState = new PnpInferenceState(cg)
  def init(model: PnpModel): PnpInferenceState = init(model.getComputationGraph())
}