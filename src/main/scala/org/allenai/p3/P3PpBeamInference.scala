package org.allenai.p3

import scala.collection.JavaConverters._
import scala.collection.mutable.MapBuilder

import org.allenai.pnp.Env
import org.allenai.pnp.PpBeamMarginals

import com.google.common.collect.HashMultimap
import com.google.common.collect.Multimap
import com.jayantkrish.jklol.ccg.CcgInference
import com.jayantkrish.jklol.ccg.CcgParse
import com.jayantkrish.jklol.ccg.chart.ChartCost
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.ccg.lambda2.ExpressionSimplifier
import com.jayantkrish.jklol.nlpannotation.AnnotatedSentence
import com.jayantkrish.jklol.training.LogFunction
import com.jayantkrish.jklol.training.NullLogFunction
import com.jayantkrish.jklol.util.CountAccumulator

/** Beam search inference algorithm for P3. This algorithm
  * performs a beam search over CCG parses using ccgInference,
  * then executes the top numLogicalForms logical forms with
  * a beam size of evalBeamSize.
  */
class P3PpBeamInference(
    val ccgInference: CcgInference, val simplifier: ExpressionSimplifier,
    val numLogicalForms: Int, val evalBeamSize: Int
) {

  def beamSearch(model: P3PpModel, sentence: AnnotatedSentence,
    env: Env): P3PpBeamMarginals[AnyRef] = {
    beamSearch(model, sentence, env, null, (x: Env) => 0.0, new NullLogFunction())
  }

  def beamSearch(model: P3PpModel, sentence: AnnotatedSentence,
    env: Env, log: LogFunction): P3PpBeamMarginals[AnyRef] = {
    beamSearch(model, sentence, env, null, (x: Env) => 0.0, log)
  }

  def beamSearch(model: P3PpModel, sentence: AnnotatedSentence, env: Env, chartCost: ChartCost,
    ppCost: Env => Double, log: LogFunction): P3PpBeamMarginals[AnyRef] = {
    log.startTimer("p3_beam/ccg_parse")
    val ccgParses = ccgInference.beamSearch(model.parser, sentence, chartCost, log).asScala
    log.stopTimer("p3_beam/ccg_parse")

    log.startTimer("p3_beam/aggregate_lf")
    val lfMap = HashMultimap.create[Expression2, CcgParse]
    val lfProbs = CountAccumulator.create[Expression2]
    aggregateParsesByLogicalForm(ccgParses, lfMap, lfProbs, simplifier)

    val sortedLfs = lfProbs.getSortedKeys().asScala
    log.stopTimer("p3_beam/aggregate_lf")

    var numEvaluated = 0
    val compGraph = model.pp.getInitialComputationGraph
    val executionMap = new MapBuilder[Expression2, PpBeamMarginals[AnyRef], Map[Expression2, PpBeamMarginals[AnyRef]]](Map())

    for (lf <- sortedLfs) {
      if (numEvaluated < numLogicalForms) {
        log.startTimer("p3_beam/eval")
        val pp = model.lfToPp(lf)
        val ppMarginals = pp.beamSearch(evalBeamSize, env, ppCost, compGraph, log)
        val executions = ppMarginals.executions

        if (executions.length > 1) {
          numEvaluated += 1
        }
        executionMap += (lf -> ppMarginals)
        log.stopTimer("p3_beam/eval")
      }
    }

    new P3PpBeamMarginals(lfMap, executionMap.result)
  }

  def aggregateParsesByLogicalForm(parses: Seq[CcgParse], map: Multimap[Expression2, CcgParse],
    probs: CountAccumulator[Expression2], simplifier: ExpressionSimplifier): Unit = {
    for (parse <- parses) {
      var lf = parse.getLogicalForm()
      if (lf != null) {
        lf = simplifier.apply(lf)
      }
      map.put(lf, parse)
      probs.increment(lf, parse.getSubtreeProbability())
    }
  }
}