package org.allenai.pnp

import com.jayantkrish.jklol.tensor.Tensor
import com.jayantkrish.jklol.util.IndexedList

/** A neural probabilistic program model consisting
  * of a collection of named Tensor parameters. These
  * parameters are used to initialize the computation
  * graph of a program during inference.
  */
class PpModel(val names: IndexedList[String], val values: List[Tensor]) extends Serializable {

  def getParameter(name: String): Tensor = {
    values(names.getIndex(name))
  }

  def getParameterIndex(name: String): Int = {
    names.getIndex(name)
  }

  def getInitialComputationGraph(): CompGraph = {
    new CompGraph(names, values.toArray)
  }
}