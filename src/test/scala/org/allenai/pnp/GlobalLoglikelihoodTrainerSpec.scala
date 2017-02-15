package org.allenai.pnp

import scala.collection.JavaConverters._
import org.scalatest._
import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._
import org.allenai.pnp.examples.DynetScalaHelpers._
import com.jayantkrish.jklol.util.IndexedList
import com.jayantkrish.jklol.training.NullLogFunction

class GlobalLoglikelihoodTrainerSpec extends FlatSpec with Matchers {
  
  myInitialize()

  val TOLERANCE = 0.01

  "GlobalLoglikelihoodTrainer" should "train" in {
    val vocab = Array(0,1,2)
    
    def lm(k: Int): Pp[Array[Int]] = {
      if (k == 1) {
        for {
          params <- Pp.param("start")
          choice <- Pp.choose(vocab, params, k - 1)
        } yield {
          Array(choice)
        }
      } else {
        for {
          rest <- lm(k - 1)
          previous = rest.last
          transition <- Pp.param("transition")
          params = pickrange(transition, previous * vocab.length, (previous + 1) * vocab.length)
          choice <- Pp.choose(vocab, params, k - 1)
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
    
    val model = PpModel.init(false)
    val startParam = model.addParameter("start", Seq(vocab.length))
    val transitionParam = model.addParameter("transition", Seq(vocab.length * vocab.length))

    val examples = List(
        PpExample(lm(3), lm(3), Env.init, makeOracle(Array(0,1,0))),
        PpExample(lm(3), lm(3), Env.init, makeOracle(Array(0,1,2)))
    )

    val sgd = new SimpleSGDTrainer(model.model, 0.1f, 0.1f)
    val trainer = new GlobalLoglikelihoodTrainer(1000, 100, model, sgd, new NullLogFunction())
    
    trainer.train(examples)
  }
}