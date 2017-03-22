package org.allenai.pnp

import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.dynet._

/** Computation graph of a neural network.
  */
class CompGraph(val model: Model,
    val paramNames: Map[String, Parameter], val lookupParamNames: Map[String, LookupParameter], 
    val locallyNormalized: Boolean) {

  def getParameter(name: String): Parameter = {
    paramNames(name)
  }

  def getLookupParameter(name: String): LookupParameter = {
    lookupParamNames(name)
  }
}

object CompGraph {
  def empty(model: Model): CompGraph = {
    new CompGraph(model, Map(), Map(), false)
  }
}