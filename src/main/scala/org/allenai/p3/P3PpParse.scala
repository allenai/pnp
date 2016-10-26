package org.allenai.p3

import org.allenai.pnp.Env

import com.jayantkrish.jklol.ccg.CcgParse
import com.jayantkrish.jklol.ccg.lambda2.Expression2

/** A CCG parse and an execution of its logical form.
  */
class P3PpParse[T](val parse: CcgParse, val lf: Expression2,
    val value: T, val env: Env, val prob: Double) {

  override def toString: String = {
    "P3PpParse(" + value + ", " + lf + ", " + prob + ")"
  }
}