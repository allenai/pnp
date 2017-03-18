package org.allenai.pnp

import edu.cmu.dynet._
import org.scalatest._

class SampleSpec extends FlatSpec with Matchers {

  "Pnp" should "sample unconditionally" in {

    val flip = Pnp.choose(Seq("a", "b"), Seq(0.25d, 0.75d))

    val samples = for (i <- 1 to 10000) yield flip.sample()

    val numA = samples.map(_.value).filter(_ == "a").size
    val numB = samples.map(_.value).filter(_ == "b").size

    numA + numB shouldBe 10000

    numA should be > 2000
    numA should be < 3000

    numB should be > 7000
    numB should be < 8000
  }

  it should "take scores into account" in {
    val flip = Pnp.chooseTag(Seq("a", "b"), "choice")

    val score = new ExecutionScore() {
      def apply(tag: Any, choice: Any, env: Env): Double = {
        if (tag == "choice" && choice == "a") 1.0 else 0.0
      }
    }

    val env = Env.init
    val inferenceState = PnpInferenceState.init.addExecutionScore(score)
    val samples = for (i <- 1 to 10000) yield flip.sample(env=env, inferenceState=inferenceState)

    // This is how the probabilities should work out.
    val aProb = math.E / (1 + math.E)
    val bProb = 1 / (1 + math.E)

    val numA = samples.map(_.value).filter(_ == "a").size
    val numB = samples.map(_.value).filter(_ == "b").size

    numA + numB shouldBe 10000

    numA.toDouble / 10000 shouldBe aProb +- 0.05
    numB.toDouble / 10000 shouldBe bProb +- 0.05
  }

  it should "take scores into account when sampling from expressions" in {
    Initialize.initialize()
    ComputationGraph.renew()

    val weightVector = new FloatVector(Seq(0f, 0f))
    val weights = Expression.input(Dim(2), weightVector)
    val flip = Pnp.choose(Array("a", "b"), weights, "choice")

    val score = new ExecutionScore() {
      def apply(tag: Any, choice: Any, env: Env): Double = {
        if (tag == "choice" && choice == "a") 1.0 else 0.0
      }
    }

    val model = new Model()
    val cg = CompGraph.empty(model)
    val inferenceState = PnpInferenceState.init(cg).addExecutionScore(score)
    val env = Env.init

    val samples = for (i <- 1 to 10000) yield flip.sample(env=env, inferenceState=inferenceState)

    // This is how the probabilities should work out.
    val aProb = math.E / (1 + math.E)
    val bProb = 1 / (1 + math.E)

    val numA = samples.map(_.value).filter(_ == "a").size
    val numB = samples.map(_.value).filter(_ == "b").size

    numA + numB shouldBe 10000

    numA.toDouble / 10000 shouldBe aProb +- 0.05
    numB.toDouble / 10000 shouldBe bProb +- 0.05
  }

  it should "sample multistep" in {

    val flip = for {
      first <- Pnp.choose(Seq(true, false), Seq(0.2d, 0.8d))
      second <- if (first) {
        Pnp.value(1)
      } else {
        Pnp.choose(Seq(3, 4), Seq(0.4d, 0.6d))
      }
      third <- if (second == 4) {
        Pnp.choose(Seq(4, 5), Seq(0.1d, 0.9d))
      } else {
        Pnp.value(second)
      }
    } yield third


    val samples = for (i <- 1 to 10000) yield flip.sample()

    val counts = samples.map(_.value).groupBy(identity).mapValues(_.size)
    counts.keySet shouldBe Set(1, 3, 4, 5)
    counts.values.sum shouldBe 10000

    // 1 = 10000 * .2  = 2000
    counts(1) should be > 1800
    counts(1) should be < 2200

    // 3 = 10000 * .8 * .4 = 3200
    counts(3) should be > 3000
    counts(3) should be < 3400

    // 4 = 10000 * .8 * .6 * .1 = 480
    counts(4) should be > 400
    counts(4) should be < 560

    // 5 = 10000 * .8 * .6 * .9 = 4320
    counts(5) should be > 4120
    counts(5) should be < 4520
  }

  type RandomVariable[T] = () => T
  class Distribution(rv: RandomVariable[Float]) extends Pnp[Float] {
    override def searchStep[C](env: Env, inferenceState: PnpInferenceState, continuation: PnpContinuation[Float, C],
        queue: PnpSearchQueue[C], finished: PnpSearchQueue[C]): Unit = ???

    /** Implements a single step of forward sampling.
      */
    override def sampleStep[C](env: Env, inferenceState: PnpInferenceState, continuation: PnpContinuation[Float, C],
        queue: PnpSearchQueue[C], finished: PnpSearchQueue[C]): Unit = {
      continuation.sampleStep(rv(), env, inferenceState, queue, finished)
    }
  }

  def uniform(lo: Float, hi: Float): Pnp[Float] = new Distribution(
    () => scala.util.Random.nextFloat * (hi - lo) + lo)

  it should "deal with 'continuous' variables" in {
    val dist = uniform(0f, 1f)

    val samples = for (i <- 1 to 10000) yield dist.sample().value

    samples.max should be <= 1f
    samples.min should be >= 0f

    val deciles = samples.groupBy(v => math.floor(10f * v)).mapValues(_.size)

    deciles.keySet shouldBe (0 until 10).toSet
    for (i <- 0 until 10) {
      deciles(i) should be > 900
      deciles(i) should be < 1100
    }
  }
}
