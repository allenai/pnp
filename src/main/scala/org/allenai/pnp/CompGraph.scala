package org.allenai.pnp

import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._

/** Computation graph of a neural network.
  */
class CompGraph(val cg: ComputationGraph, val model: Model,
    val paramNames: Map[String, Parameter], val lookupParamNames: Map[String, LookupParameter], 
    val locallyNormalized: Boolean) {
  
  // Initialize the nodes of the graph with a node per
  // parameter.
  val paramExpressions = new Array[Expression](paramNames.size)
  for (i <- 0 until paramNames.size) {
    paramExpressions(i) = parameter(cg, new Parameter(model, i)) 
  }

  def getParameter(name: String): Parameter = {
    paramNames(name)
  }

  def getLookupParameter(name: String): LookupParameter = {
    lookupParamNames(name)
  }
}

object CompGraph {
  def empty(cg: ComputationGraph, model: Model): CompGraph = {
    new CompGraph(cg, model, Map(), Map(), false)
  }
}