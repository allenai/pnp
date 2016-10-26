package org.allenai.p3

import java.util.Arrays

import scala.collection.JavaConverters._

import org.allenai.pnp.ParametricPpModel
import org.allenai.pnp.Pp

import com.jayantkrish.jklol.ccg.ParametricCcgParser
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.models.parametric.ListSufficientStatistics
import com.jayantkrish.jklol.models.parametric.ParametricFamily
import com.jayantkrish.jklol.models.parametric.SufficientStatistics
import com.jayantkrish.jklol.nlpannotation.AnnotatedSentence
import com.jayantkrish.jklol.training.LogFunction
import com.jayantkrish.jklol.training.NullLogFunction

/** A parametric family of P3 models. The family is composed of
  * a family of CCG parsers and a family of neural probabilistic
  * program models. The family also contains a conversion function
  * for mapping logical forms represented as Expression2 to
  * neural probabilistic programs represented as Pp.
  */
class ParametricP3PpModel(
    val ccgFamily: ParametricCcgParser, val evalFamily: ParametricPpModel,
    val lfToPp: Expression2 => Pp[AnyRef]
) extends ParametricFamily[P3PpModel] {
  val serialVersionUID = 1L

  val CCG_PARAMETER_NAME = "ccg"
  val EVAL_PARAMETER_NAME = "eval"

  override def getNewSufficientStatistics(): SufficientStatistics = {
    new ListSufficientStatistics(
      Arrays.asList(CCG_PARAMETER_NAME, EVAL_PARAMETER_NAME),
      Arrays.asList(
        ccgFamily.getNewSufficientStatistics(),
        evalFamily.getNewSufficientStatistics()
      )
    )
  }

  override def getModelFromParameters(parameters: SufficientStatistics): P3PpModel = {
    val parameterList = parameters.coerceToList().getStatistics()
    val parser = ccgFamily.getModelFromParameters(parameterList.get(0))
    val ppModel = evalFamily.getModelFromParameters(parameterList.get(1))

    new P3PpModel(parser, ppModel, lfToPp)
  }

  override def getParameterDescription(parameters: SufficientStatistics): String = {
    getParameterDescription(parameters, -1)
  }

  override def getParameterDescription(parameters: SufficientStatistics, numFeatures: Int): String = {
    val parameterList = parameters.coerceToList().getStatistics()

    val sb = new StringBuilder()
    sb.append(ccgFamily.getParameterDescription(parameterList.get(0), numFeatures))
    sb.append(evalFamily.getParameterDescription(parameterList.get(1), numFeatures))

    sb.toString()
  }

  def incrementSufficientStatistics[A](
    gradient: SufficientStatistics,
    currentParameters: SufficientStatistics, sentence: AnnotatedSentence,
    marginals: P3PpBeamMarginals[A], count: Double
  ): Unit = {
    incrementSufficientStatistics(gradient, currentParameters, sentence, marginals, count,
      new NullLogFunction())
  }

  def incrementSufficientStatistics[A](
    gradient: SufficientStatistics,
    currentParameters: SufficientStatistics, sentence: AnnotatedSentence,
    marginals: P3PpBeamMarginals[A], count: Double, logFunction: LogFunction
  ): Unit = {
    val gradientList = gradient.coerceToList().getStatistics()
    val parameterList = currentParameters.coerceToList().getStatistics()

    for (lf <- marginals.executions.keys) {
      logFunction.startTimer("p3increment/execution_partition")
      val executionMarginals = marginals.executions(lf)
      logFunction.stopTimer("p3increment/execution_partition")

      logFunction.startTimer("p3increment/parse_increment")
      val executionPartitionFunction = executionMarginals.partitionFunction
      for (parse <- marginals.lfMap.get(lf).asScala) {
        ccgFamily.incrementSufficientStatistics(gradientList.get(0), parameterList.get(0),
          sentence, parse, parse.getSubtreeProbability * executionPartitionFunction * count)
      }
      logFunction.stopTimer("p3increment/parse_increment")

      logFunction.startTimer("p3increment/parse_partition")
      val parsePartitionFunction = marginals.lfMap.get(lf).asScala
        .map(x => x.getSubtreeProbability).sum
      logFunction.stopTimer("p3increment/parse_partition")

      logFunction.startTimer("p3increment/execution_increment")
      evalFamily.incrementSufficientStatistics(gradientList.get(1), executionMarginals,
        parsePartitionFunction * count, logFunction)
      logFunction.stopTimer("p3increment/execution_increment")
    }
  }
}

