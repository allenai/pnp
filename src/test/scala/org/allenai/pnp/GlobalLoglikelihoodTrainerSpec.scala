package org.allenai.pnp

import org.allenai.pnp.ExecutionScore.ExecutionScore
import org.scalatest.FlatSpec
import org.scalatest.Matchers

import com.jayantkrish.jklol.training.NullLogFunction

import edu.cmu.dynet._
import com.jayantkrish.jklol.util.IndexedList
import com.jayantkrish.jklol.training.NullLogFunction

class GlobalLoglikelihoodTrainerSpec extends FlatSpec with Matchers {
    
  Initialize.initialize()

  val TOLERANCE = 0.01

  "GlobalLoglikelihoodTrainer" should "train" in {
    val vocab = Array(0,1,2)
    
    def lm(k: Int): Pnp[Array[Int]] = {
      if (k == 1) {
        for {
          params <- Pnp.param("start")
          choice <- Pnp.choose(vocab, params, k - 1)
        } yield {
          Array(choice)
        }
      } else {
        for {
          rest <- lm(k - 1)
          previous = rest.last
          transition <- Pnp.param("transition")
          params = Expression.pickrange(
            transition, previous * vocab.length, (previous + 1) * vocab.length)
          choice <- Pnp.choose(vocab, params, k - 1)
        } yield {
          rest ++ Array(choice)
        }
      }
    }

    def makeOracle(label: Array[Int]): ExecutionScore = {
      new ExecutionScore() {
        def apply(tag: Any, choice: Any, env: Env): Double = {
          if (tag != null && tag.isInstanceOf[Int]) {
            val tagInt = tag.asInstanceOf[Int]
            if (tagInt >= 0 && tagInt < label.length) {
              if (choice == label(tagInt)) {
                0.0
              } else {
                Double.NegativeInfinity
              }
            } else {
              Double.NegativeInfinity
            }
          } else {
            0.0
          }
        }
      }
    }
    
    val model = PnpModel.init(false)
    val startParam = model.addParameter("start", Dim(vocab.length))
    val transitionParam = model.addParameter("transition", Dim(vocab.length * vocab.length))

    val examples = List(
        PnpExample(lm(3), lm(3), Env.init, makeOracle(Array(0,1,0))),
        PnpExample(lm(3), lm(3), Env.init, makeOracle(Array(0,1,2)))
    )

    val sgd = new SimpleSGDTrainer(model.model, 0.1f, 0.1f)
    val trainer = new GlobalLoglikelihoodTrainer(1000, 100, -1, model, sgd, new NullLogFunction())
    // val trainer = new BsoTrainer(100, 1, -1, model, sgd, new NullLogFunction())
    
    trainer.train(examples)
  }
}
