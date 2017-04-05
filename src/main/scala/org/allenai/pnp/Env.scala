package org.allenai.pnp

import com.jayantkrish.jklol.util.IndexedList
import com.jayantkrish.jklol.training.LogFunction
import com.jayantkrish.jklol.training.NullLogFunction
import edu.cmu.dynet._

/** Mutable global state of a neural probabilistic program
  * execution. Env also tracks the chosen values for any
  * nondeterministic choices whose values depended on
  * computation graph nodes. These values are necessary
  * to compute gradients with respect to the neural network
  * parameters.
  *
  * Env is immutable.
  */
class Env(val labels: List[Int], val labelNodeIds: List[Expression],
    varnames: IndexedList[String], vars: Array[Any]) {

  /** Get the value of the named variable as an instance
    * of type A.
    */
  def getVar[A](name: String): A = {
    vars(varnames.getIndex(name)).asInstanceOf[A]
  }
  
  def getVar[A](name: String, default: A): A = {
    if (varnames.contains(name)) {
      getVar(name)
    } else {
      default
    }
  }

  def getVar[A](nameInt: Int): A = {
    vars(nameInt).asInstanceOf[A]
  }

  def getVar[A](nameInt: Int, default: A): A = {
    if (nameInt < vars.length) {
      getVar(nameInt)
    } else {
      default
    }
  }

  /** Get a new environment with the named variable
    * set to value.
    */
  def setVar(name: String, value: Any): Env = {
    val nextVarNames = if (varnames.contains(name)) {
      varnames
    } else {
      val i = IndexedList.create(varnames)
      i.add(name)
      i
    }

    val nextVars = Array.ofDim[Any](nextVarNames.size)
    Array.copy(vars, 0, nextVars, 0, vars.size)
    val index = nextVarNames.getIndex(name)
    nextVars(index) = value

    new Env(labels, labelNodeIds, nextVarNames, nextVars)
  }

  def setVar(nameInt: Int, value: Any): Env = {
    val nextVars = Array.ofDim[Any](vars.size)
    Array.copy(vars, 0, nextVars, 0, vars.size)
    nextVars(nameInt) = value

    new Env(labels, labelNodeIds, varnames, nextVars)
  }

  def isVarBound(name: String): Boolean = {
    varnames.contains(name)
  }

  /** Attaches a label to a node of the computation graph in this
    * execution.
    */
  def addLabel(param: Expression, index: Int): Env = {
    new Env(index :: labels, param :: labelNodeIds, varnames, vars)
  }
  
  /** Get a scalar-valued expression that evaluates to the
    * score of the execution that this env is part of. If   
    * normalize is false, this score is computed by summing
    * the scores associated with choice. If normalize is true,
    * the score is computed by summing the negative log-softmax
    * scores of each choice.
    */
  def getScore: Expression = {
    var exScore = Expression.input(0)
    for ((expr, labelInd) <- labelNodeIds.zip(labels)) {
      val decisionScore = Expression.pick(expr, labelInd)
      exScore = exScore + decisionScore
    }
    exScore
  }
}

object Env {
  def init: Env = {
    new Env(List.empty, List.empty, IndexedList.create(), Array())
  }
}