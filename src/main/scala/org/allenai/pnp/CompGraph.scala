package org.allenai.pnp

import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._

/** Computation graph of a neural network.
  */
class CompGraph(val cg: ComputationGraph, val model: Model,
    val paramNames: IndexedList[String], val lookupParamNames: IndexedList[String], 
    val locallyNormalized: Boolean) {
  
  // Initialize the nodes of the graph with a node per
  // parameter.
  val paramExpressions = new Array[Expression](paramNames.size)
  for (i <- 0 until paramNames.size) {
    paramExpressions(i) = parameter(cg, new Parameter(model, i)) 
  }

  def getParameter(name: String): Expression = {
    paramExpressions(paramNames.getIndex(name))
  }
  
  def getLookupParameter(name: String): LookupParameter = {
    new LookupParameter(model, lookupParamNames.getIndex(name))
  }
}

object CompGraph {
  def empty(cg: ComputationGraph, model: Model): CompGraph = {
    new CompGraph(cg, model, IndexedList.create[String], 
        IndexedList.create[String], false)
  }
}

