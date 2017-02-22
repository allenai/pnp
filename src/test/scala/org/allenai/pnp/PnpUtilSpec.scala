package org.allenai.pnp

import scala.collection.JavaConverters._

import org.scalatest._
import org.scalatest.Matchers

import com.jayantkrish.jklol.ccg.lambda.ExpressionParser

class PnpUtilSpec extends FlatSpec with Matchers {

  val TOLERANCE = 0.0001
  val parser = ExpressionParser.expression2

  def flip(p: Double): Pnp[Boolean] = {
    Pnp.chooseMap(Seq((true, p), (false, 1.0 - p)))
  }

  val bindings = Map[String, AnyRef](
    "true" -> true.asInstanceOf[AnyRef],
    "false" -> false.asInstanceOf[AnyRef],
    "coin" -> Pnp.chooseMap(Seq((true, 0.6), (false, 0.4))),
    "flipProb" -> 0.6.asInstanceOf[AnyRef],
    "flipProb2" -> 0.55.asInstanceOf[AnyRef],
    "flip" -> PnpUtil.wrap(flip _),
    "filter" -> PnpUtil.wrap(PnpUtil.filter _),
    "list" -> { x: Vector[AnyRef] => Pnp.value(x.toList) },
    "concat" -> PnpUtil.wrap2({ (x: String, y: String) => x ++ y })
  )

  def runTest[A](exprString: String, expected: Seq[(A, Double)]): Unit = {
    val expr = parser.parse(exprString)
    val pp = PnpUtil.lfToPnp(expr, bindings)

    val values = pp.beamSearch(100)

    for ((value, expected) <- values.zip(expected)) {
      value._1 should be(expected._1)
      value._2 should be(expected._2 +- TOLERANCE)
    }
  }

  "PpUtil" should "correctly interpret constants" in {
    runTest("coin", Seq((true, 0.6), (false, 0.4)))
  }

  it should "correctly interpret string constants" in {
    runTest("\"foo\"", Seq(("foo", 1.0)))
  }

  it should "correctly interpret applications" in {
    runTest("(flip flipProb)", Seq((true, 0.6), (false, 0.4)))
  }

  it should "correctly interpret applications (2)" in {
    runTest("(list flipProb)", Seq((List(0.6), 1.0)))
  }

  it should "correctly interpret applications (3)" in {
    runTest("(concat \"foo\" \"bar\")", Seq(("foobar", 1.0)))
  }

  it should "correctly interpret filters" in {
    runTest(
      "(filter (lambda (x) (flip x)) (list flipProb flipProb2))",
      Seq((List(0.6, 0.55), 0.6 * 0.55), (List(0.6), 0.6 * 0.45),
        (List(0.55), 0.4 * 0.55), (List(), 0.4 * 0.45))
    )
  }
}