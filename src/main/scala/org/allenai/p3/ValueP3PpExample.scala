package org.allenai.p3

import org.allenai.pnp.Env

import com.jayantkrish.jklol.ccg.chart.ChartCost
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.nlpannotation.AnnotatedSentence

class ValueP3PpExample(sentence: AnnotatedSentence, env: Env,
  chartCost: ChartCost, executionCost: Env => Double, val label: AnyRef)
    extends P3PpExample(sentence, env, chartCost, executionCost) {

  override def lfFilter(expression: Expression2): Boolean = {
    true
  }

  override def executionFilter(denotation: AnyRef, env: Env): Boolean = {
    denotation == label
  }
}