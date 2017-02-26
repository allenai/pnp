package org.allenai.pnp

import com.jayantkrish.jklol.ccg.lambda2.StaticAnalysis
import com.jayantkrish.jklol.ccg.lambda2.Expression2

import scala.collection.JavaConverters._
import com.google.common.base.Preconditions

/** Utilities for converting logical forms represented using
  * {@code Expression2} to neural probabilistic programs.
  */
object PnpUtil {

  /** Convert {@code lf} to a neural probabilistic program.
    * {@code bindings} represents the environment in which
    * {@code lf} is evaluated; it maps names in {@code lf} to
    * their corresponding probabilistic program values. This
    * function fails if a name in {@code lf} is not contained
    * in {@code bindings}.
    *
    * Non-function values in bindings can be of any type.
    * Functions must have type Vector[AnyRef] => Pnp[AnyRef].
    * The wrap functions below can be used to conveniently
    * convert existing functions to this type.
    */
  def lfToPnp(lf: Expression2, bindings: Map[String, AnyRef]): Pnp[AnyRef] = {
    if (lf.isConstant()) {
      if (lf.isStringValue()) {
        Pnp.value(lf.getStringValue)
      } else {
        // Look up the constant's value in bindings.
        val valueOption = bindings.get(lf.getConstant)
        Preconditions.checkState(valueOption.isDefined, "Unbound variable: %s", lf.getConstant)

        val value = valueOption.get
        if (value.isInstanceOf[Pnp[_]]) {
          value.asInstanceOf[Pnp[AnyRef]]
        } else {
          // Wrap non-Pnp values to guarantee that every
          // expression evaluates to a Pnp[AnyRef].
          Pnp.value(value)
        }
      }
    } else if (StaticAnalysis.isLambda(lf)) {
      // Create a Scala function representing the lambda. 
      val args = StaticAnalysis.getLambdaArguments(lf).asScala
      val body = StaticAnalysis.getLambdaBody(lf)

      def lambdaValue(argList: Vector[AnyRef]): Pnp[AnyRef] = {
        val newBindings = bindings ++ args.zip(argList)
        lfToPnp(body, newBindings)
      }
      Pnp.value(lambdaValue _)
    } else {
      // Function application.
      // Generate the distributions over values for the function
      // and each of its arguments. 
      val subexpressionValues = lf.getSubexpressions.asScala.map(x => lfToPnp(x, bindings))
      val subexpressionListPnp = subexpressionValues.foldLeft(Pnp.value(Vector[AnyRef]()))(
        (vecPnp, valPnp) => for {
          x <- vecPnp
          y <- valPnp
        } yield {
          x :+ y
        }
      )

      // Apply each possible function to its arguments. 
      for {
        valueList <- subexpressionListPnp
        args = valueList.slice(1, valueList.size)
        numArgs = args.size
        func = valueList(0)

        value <- func.asInstanceOf[AnyRef => Pnp[AnyRef]].apply(args)
      } yield {
        value
      }
    }
  }

  def wrap[A, P](f: A => Pnp[P]): (Vector[AnyRef] => Pnp[P]) = {
    x: Vector[AnyRef] =>
      {
        Preconditions.checkArgument(
          x.size == 1,
          "Wrong number of arguments. Expected 1 got %s", x.size.asInstanceOf[AnyRef]
        )
        f(x(0).asInstanceOf[A])
      }
  }

  def wrap[A, B, P](f: (A, B) => Pnp[P]): (Vector[AnyRef] => Pnp[P]) = {
    x: Vector[AnyRef] =>
      {
        Preconditions.checkArgument(
          x.size == 2,
          "Wrong number of arguments. Expected 2 got %s", x.size.asInstanceOf[AnyRef]
        )
        f(x(0).asInstanceOf[A], x(1).asInstanceOf[B])
      }
  }

  def wrap2[A, P](f: A => P): (Vector[AnyRef] => Pnp[P]) = {
    x: Vector[AnyRef] =>
      {
        Preconditions.checkArgument(
          x.size == 1,
          "Wrong number of arguments. Expected 1 got %s", x.size.asInstanceOf[AnyRef]
        )
        Pnp.value(f(x(0).asInstanceOf[A]))
      }
  }

  def wrap2[A, B, P](f: (A, B) => P): (Vector[AnyRef] => Pnp[P]) = {
    x: Vector[AnyRef] =>
      {
        Preconditions.checkArgument(
          x.size == 2,
          "Wrong number of arguments. Expected 2 got %s", x.size.asInstanceOf[AnyRef]
        )
        Pnp.value(f(x(0).asInstanceOf[A], x(1).asInstanceOf[B]))
      }
  }

  def filter[A](f: AnyRef => Pnp[Boolean], elts: List[A]): Pnp[List[A]] = {
    elts.foldRight(Pnp.value(List[A]()))(
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
  
  // TODO: make this work for any seq type.
  def map[A,B](f: A => Pnp[B], elts: List[A]): Pnp[List[B]] = {
    elts.foldRight(Pnp.value(List[B]()))(
      (elt, list) => for {
        mapped <- f(elt)
        l <- list
      } yield {
        mapped :: l
      }
    )
  }
}