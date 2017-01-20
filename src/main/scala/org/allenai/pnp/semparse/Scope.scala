package org.allenai.pnp.semparse

import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.ccg.lambda.Type

/** A list of typed variables that are in scope. These
  * variables are bound in lambda expressions that contain
  * the current expression.
  */
case class Scope(val vars: List[(Expression2, Type)]) {

  def getVariableExpressions(t: Type): List[Expression2] = {
    vars.filter(_._2.equals(t)).map(_._1)
  }
  
  def getVariableTemplates(t: Type): List[Template] = {
    getVariableExpressions(t).map(x => ConstantTemplate(t, x))
  }
  
  /** Extend this scope with additional variables with the
    * corresponding types.
    */
  def extend(types: List[Type]): (Scope, List[String]) = {
    var varNames = List[String]()
    var nextVars = vars
    for (t <- types) {
      val varName = "$" + nextVars.size
      varNames = varName :: varNames
      nextVars = (Expression2.constant(varName), t) :: nextVars
    }

    val nextScope = new Scope(nextVars)
    (nextScope, varNames)
  }
}
