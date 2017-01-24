package org.allenai.pnp

import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._

/** Computation graph of a neural network.
  */
class CompGraph(val cg: ComputationGraph, val model: Model,
    val paramNames: IndexedList[String], val params: Array[Parameter],
    val lookupParamNames: IndexedList[String], val lookupParams: Array[LookupParameter],
    val locallyNormalized: Boolean) {
  
  // Initialize the nodes of the graph with a node per
  // parameter.
  val paramExpressions = new Array[Expression](params.length)
  for (i <- 0 until params.length) {
    paramExpressions(i) = parameter(cg, params(i)) 
  }

  def getParameter(name: String): Expression = {
    paramExpressions(paramNames.getIndex(name))
  }
  
  def getLookupParameter(name: String): LookupParameter = {
    lookupParams(lookupParamNames.getIndex(name))
  }
}

object CompGraph {
  def empty(cg: ComputationGraph, model: Model): CompGraph = {
    new CompGraph(cg, model, IndexedList.create[String], Array(),
        IndexedList.create[String], Array(), false)
  }
}

