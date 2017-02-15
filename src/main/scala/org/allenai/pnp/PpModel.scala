package org.allenai.pnp

import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.dynet._

/** A neural probabilistic program model consisting
  * of a collection of named Tensor parameters. These
  * parameters are used to initialize the computation
  * graph of a program during inference.
  */
class PpModel(var names: Map[String, Parameter], var lookupNames: Map[String, LookupParameter], 
    val model: Model, val locallyNormalized: Boolean) {

  def addParameter(name: String, dim: Dim): Parameter = {
    val param = model.add_parameters(dim)
    names += (name -> param)
    param
  }
  
  def addLookupParameter(name: String, lookupNum: Long, dim: Dim): LookupParameter = {
    val param = model.add_lookup_parameters(lookupNum, dim)
    lookupNames += (name -> param)
    param
  }

  def getParameter(name: String): Parameter = {
    names(name)
  }

  def getLookupParameter(name: String): LookupParameter = {
    lookupNames(name)
  }

  def getComputationGraph(cg: ComputationGraph): CompGraph = {
    new CompGraph(cg, model, names, lookupNames, locallyNormalized)
  }
}

object PpModel {
  def init(locallyNormalized: Boolean): PpModel = {
    new PpModel(Map(), Map(), new Model, locallyNormalized)
  }
}