package org.allenai.pnp

import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.dynet._

/** A neural probabilistic program model consisting
  * of a collection of named Tensor parameters. These
  * parameters are used to initialize the computation
  * graph of a program during inference.
  */
class PpModel(val names: IndexedList[String], val parameters: Array[Parameter],
    val lookupNames: IndexedList[String], val lookupParameters: Array[LookupParameter],
    val model: Model, val locallyNormalized: Boolean) extends Serializable {

  def getParameterIndex(name: String): Int = {
    names.getIndex(name)
  }
  
  def getLookupParameterIndex(name: String): Int = {
    lookupNames.getIndex(name)
  }

  def getInitialComputationGraph(cg: ComputationGraph): CompGraph = {
    new CompGraph(cg, model, names, parameters,
        lookupNames, lookupParameters, locallyNormalized)
  }
}