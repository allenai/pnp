package org.allenai.pnp

import scala.collection.JavaConverters._

import com.jayantkrish.jklol.models.VariableNumMap
import com.jayantkrish.jklol.models.parametric.ListSufficientStatistics
import com.jayantkrish.jklol.models.parametric.ParametricFamily
import com.jayantkrish.jklol.models.parametric.SufficientStatistics
import com.jayantkrish.jklol.models.parametric.TensorSufficientStatistics
import com.jayantkrish.jklol.util.IndexedList
import com.jayantkrish.jklol.training.LogFunction
import com.jayantkrish.jklol.training.NullLogFunction

/** A parametric family of neural probabilistic programs. This class
  * is a collection of named neural network parameters and
  * their expected dimensionalities. Instantiating this class with
  * a parameter vector returns a {@code PpModel}, which represents
  * a particular neural network within this family.
  */
class ParametricPpModel(paramNames: IndexedList[String], dims: List[VariableNumMap])
    extends ParametricFamily[PpModel] {

  override def getNewSufficientStatistics(): SufficientStatistics = {
    val stats = dims.map(x => TensorSufficientStatistics.createDense(x)).asInstanceOf[List[SufficientStatistics]]
    new ListSufficientStatistics(paramNames, stats.asJava)
  }

  override def getModelFromParameters(parameters: SufficientStatistics): PpModel = {
    val tensors = parameters.coerceToList().getStatistics.asScala.map(
      x => x.asInstanceOf[TensorSufficientStatistics].get
    ).toList
    new PpModel(paramNames, tensors)
  }

  def incrementSufficientStatistics[A](
    gradient: SufficientStatistics,
    marginals: PpBeamMarginals[A], count: Double
  ): Unit = {
    incrementSufficientStatistics(gradient, marginals, count, new NullLogFunction())
  }

  /** Increment the neural network parameters in gradient using
    * the program executions (and corresponding computation graph)
    * in marginals. The gradient update is the sum of the unnormalized
    * probability of each execution times its gradient computed by
    * backpropagation through the computation graph.
    */
  def incrementSufficientStatistics[A](
    gradient: SufficientStatistics,
    marginals: PpBeamMarginals[A], count: Double, log: LogFunction
  ): Unit = {
    // Initialize the computation graph's backpropagation data
    // structure with the labels from each execution.
    val graph = marginals.graph
    val backprop = graph.backprop
    var numLabels = 0
    log.startTimer("pp/initialize_backprop")
    for (execution <- marginals.executions) {
      val prob = execution.prob * count

      val labelNodeIds = execution.env.labelNodeIds
      val labels = execution.env.labels

      for ((id, label) <- labelNodeIds.zip(labels)) {
        backprop.incrementGradient(id, label, prob)
        numLabels += 1
      }
    }
    log.stopTimer("pp/initialize_backprop")

    if (numLabels == 0) {
      // Don't need to do anything if none of the executions
      // have labels, because the gradient will be 0.
      return
    }

    // Run backpropagation to compute gradients for all other
    // nodes in the graph.
    log.startTimer("pp/backprop")
    backprop.backprop()
    log.stopTimer("pp/backprop")

    log.startTimer("pp/increment_gradient")
    val gradientList = gradient.coerceToList()
    for (paramName <- graph.paramNames.items().asScala) {
      val gradient = backprop.getGradient(graph.getParameter(paramName).id)
      if (gradient != null) {
        val param = gradientList.getStatisticByName(paramName).asInstanceOf[TensorSufficientStatistics]
        param.increment(gradient, 1.0)
      }
    }
    log.stopTimer("pp/increment_gradient")
  }

  override def getParameterDescription(parameters: SufficientStatistics): String = {
    getParameterDescription(parameters, -1)
  }

  override def getParameterDescription(parameters: SufficientStatistics, num: Int): String = {
    val sb = new StringBuilder()
    val listParams = parameters.coerceToList()
    for (paramName <- paramNames.items().asScala) {
      sb.append(paramName)
      sb.append("\n")
      sb.append(listParams.getStatisticByName(paramName)
        .asInstanceOf[TensorSufficientStatistics].getFactor.getParameterDescription)
    }
    sb.toString
  }
}