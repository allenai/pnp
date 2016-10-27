package org.allenai.p3

import scala.collection.JavaConverters._

import org.allenai.pnp.Env
import org.allenai.pnp.PpBeamMarginals

import com.google.common.collect.HashMultimap
import com.jayantkrish.jklol.ccg.CcgParse
import com.jayantkrish.jklol.ccg.lambda2.Expression2

/** Marginal distribution over CCG parses of a sentence
  * and the executions of their corresponding logical forms.
  */
class P3PpBeamMarginals[T](
    val lfMap: HashMultimap[Expression2, CcgParse],
    val executions: Map[Expression2, PpBeamMarginals[T]]
) {

  /** The denominator for normalizing the unnormalized
    * probabilities of parses.
    */
  val partitionFunction = executions.map(x =>
    x._2.partitionFunction() * (lfMap.get(x._1).asScala.map(_.getSubtreeProbability).sum)).sum

  /** Condition the distribution on pred being true by
    * setting the probability of all logical forms
    * where pred is false to 0.
    */
  def conditionLfs(pred: Expression2 => Boolean): P3PpBeamMarginals[T] = {
    val newLfMap = HashMultimap.create[Expression2, CcgParse]
    for (lf <- lfMap.keys.asScala) {
      if (pred(lf)) {
        newLfMap.putAll(lf, lfMap.get(lf))
      }
    }

    new P3PpBeamMarginals(newLfMap, executions.filterKeys(pred))
  }

  /** Condition the distribution on pred being true by
    * setting the probability of all executions where
    * pred is false to 0.
    */
  def conditionExecutions(pred: (T, Env) => Boolean): P3PpBeamMarginals[T] = {
    new P3PpBeamMarginals(lfMap, executions.mapValues(x => x.condition(pred)))
  }

  /** Gets a list of parses, each of which represents a
    * CCG parse and a single execution of its logical form.
    * Each parse has an unnormalized probability that can be
    * divided by partitionFunction to get a probability.
    */
  def parses: List[P3PpParse[T]] = {
    val parses = for {
      entry <- executions.toList
      lf = entry._1
      ccg <- lfMap.get(lf).asScala
      marginals = entry._2
      execution <- marginals.executions
    } yield {
      new P3PpParse(ccg, lf, execution.value, execution.env, execution.prob * ccg.getSubtreeProbability)
    }

    parses.sortBy(_.prob).reverse
  }
}