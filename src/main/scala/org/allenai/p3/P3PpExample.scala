package org.allenai.p3

import org.allenai.pnp.Env

import com.jayantkrish.jklol.ccg.chart.ChartCost
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.nlpannotation.AnnotatedSentence

/** A training example for P3. An example has two model
  * inputs: a sentence and an environment in which its logical
  * forms will be executed. From these inputs, the model
  * predicts a CCG parse, logical form, and its execution.
  * Labels for these elements are provided using search filters,
  * with chartCost/lfFilter applying to the CCG parser and
  * executionCost/executionFilter applying to the neural probabilistic
  * program execution. The two costs are applied during inference
  * to restrict the search space, which restricts their expressivity
  * but makes training easier.
  */
abstract class P3PpExample(val sentence: AnnotatedSentence, val env: Env,
    val chartCost: ChartCost, val executionCost: Env => Double) {

  def lfFilter(expression: Expression2): Boolean

  def executionFilter(denotation: AnyRef, env: Env): Boolean
}