package org.allenai.pnp

import com.jayantkrish.jklol.ccg.lambda2.StaticAnalysis
import com.jayantkrish.jklol.ccg.lambda2.Expression2

import scala.collection.JavaConverters._
import com.google.common.base.Preconditions

/** Utilities for converting logical forms represented using
  * {@code Expression2} to neural probabilistic programs.
  */
object PpUtil {

  /** Convert {@code lf} to a neural probabilistic program.
    * {@code bindings} represents the environment in which
    * {@code lf} is evaluated; it maps names in {@code lf} to
    * their corresponding probabilistic program values. This
    * function fails if a name in {@code lf} is not contained
    * in {@code bindings}.
    *
    * Non-function values in bindings can be of any type.
    * Functions must have type Vector[AnyRef] => Pp[AnyRef].
    * The wrap functions below can be used to conveniently
    * convert existing functions to this type.
    */
  def lfToPp(lf: Expression2, bindings: Map[String, AnyRef]): Pp[AnyRef] = {
    if (lf.isConstant()) {
      if (lf.isStringValue()) {
        Pp.value(lf.getStringValue)
      } else {
        // Look up the constant's value in bindings.
        val valueOption = bindings.get(lf.getConstant)
        Preconditions.checkState(valueOption.isDefined, "Unbound variable: %s", lf.getConstant)

        val value = valueOption.get
        if (value.isInstanceOf[Pp[_]]) {
          value.asInstanceOf[Pp[AnyRef]]
        } else {
          // Wrap non-Pp values to guarantee that every
          // expression evaluates to a Pp[AnyRef].
          Pp.value(value)
        }
      }
    } else if (StaticAnalysis.isLambda(lf)) {
      // Create a Scala function representing the lambda. 
      val args = StaticAnalysis.getLambdaArguments(lf).asScala
      val body = StaticAnalysis.getLambdaBody(lf)

      def lambdaValue(argList: Vector[AnyRef]): Pp[AnyRef] = {
        val newBindings = bindings ++ args.zip(argList)
        lfToPp(body, newBindings)
      }
      Pp.value(lambdaValue _)
    } else {
      // Function application.
      // Generate the distributions over values for the function
      // and each of its arguments. 
      val subexpressionValues = lf.getSubexpressions.asScala.map(x => lfToPp(x, bindings))
      val subexpressionListPp = subexpressionValues.foldLeft(Pp.value(Vector[AnyRef]()))(
        (vecPp, valPp) => for {
          x <- vecPp
          y <- valPp
        } yield {
          x :+ y
        }
      )

      // Apply each possible function to its arguments. 
      for {
        valueList <- subexpressionListPp
        args = valueList.slice(1, valueList.size)
        numArgs = args.size
        func = valueList(0)

        value <- func.asInstanceOf[AnyRef => Pp[AnyRef]].apply(args)
      } yield {
        value
      }
    }
  }

  def wrap[A, P](f: A => Pp[P]): (Vector[AnyRef] => Pp[P]) = {
    x: Vector[AnyRef] =>
      {
        Preconditions.checkArgument(
          x.size == 1,
          "Wrong number of arguments. Expected 1 got %s", x.size.asInstanceOf[AnyRef]
        )
        f(x(0).asInstanceOf[A])
      }
  }

  def wrap[A, B, P](f: (A, B) => Pp[P]): (Vector[AnyRef] => Pp[P]) = {
    x: Vector[AnyRef] =>
      {
        Preconditions.checkArgument(
          x.size == 2,
          "Wrong number of arguments. Expected 2 got %s", x.size.asInstanceOf[AnyRef]
        )
        f(x(0).asInstanceOf[A], x(1).asInstanceOf[B])
      }
  }

  def wrap2[A, P](f: A => P): (Vector[AnyRef] => Pp[P]) = {
    x: Vector[AnyRef] =>
      {
        Preconditions.checkArgument(
          x.size == 1,
          "Wrong number of arguments. Expected 1 got %s", x.size.asInstanceOf[AnyRef]
        )
        Pp.value(f(x(0).asInstanceOf[A]))
      }
  }

  def wrap2[A, B, P](f: (A, B) => P): (Vector[AnyRef] => Pp[P]) = {
    x: Vector[AnyRef] =>
      {
        Preconditions.checkArgument(
          x.size == 2,
          "Wrong number of arguments. Expected 2 got %s", x.size.asInstanceOf[AnyRef]
        )
        Pp.value(f(x(0).asInstanceOf[A], x(1).asInstanceOf[B]))
      }
  }

  def filter[A](f: AnyRef => Pp[Boolean], elts: List[A]): Pp[List[A]] = {
    elts.foldRight(Pp.value(List[A]()))(
      (elt, list) => for {
        t <- f(Vector(elt))
        l <- list
      } yield {
        if (t) {
          elt :: l
        } else {
          l
        }
      }
    )
  }
}