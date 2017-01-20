package org.allenai.pnp.examples

import edu.cmu.dynet.{ Dim, Expression, LongVector }
import edu.cmu.dynet.dynet_swig.{ exprPlus, exprMinus, exprTimes }

import scala.language.implicitConversions

// This will go in an associated shared library
object DynetScalaHelpers {
  implicit def makeDim(dims: Seq[Int]): Dim = {
    val dimInts = new LongVector
    dims.foreach(dimInts.add)
    new Dim(dimInts)
  }

  implicit class RichExpression(e: Expression) {
    def +(e2: Expression): Expression = exprPlus(e, e2)
    def -(e2: Expression): Expression = exprMinus(e, e2)

    def *(e2: Expression): Expression = exprTimes(e, e2)
  }

}

