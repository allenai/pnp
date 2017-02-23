package org.allenai.dqa.labeling

import org.allenai.pnp.semparse.SemanticParser
import org.allenai.pnp.PnpExample
import org.allenai.pnp.Pnp
import org.allenai.pnp.Env
import org.allenai.pnp.PnpModel

class LabelingP3Model(val parser: SemanticParser,
    val executor: LabelingExecutor, val answerSelector: AnswerSelector) {
  
  def exampleToPnpExample(ex: PreprocessedLabelingExample): PnpExample[Int] = {
    val denotationDist = for {
      // TODO: stage beam search?
      lf <- parser.generateExpression(ex.tokenIds, ex.entityLinking)
      denotation <- executor.execute(lf, ex.ex.diagram)
    } yield {
      denotation
    }

    val unconditional = for {
      denotation <- denotationDist
      answer <- answerSelector.selectAnswer(denotation, ex.ex.answerOptions)
    } yield {
      answer
    }

    val conditional = for {
      denotation <- denotationDist
      // choose the answer and condition on getting the correct answer
      // in a single search step to reduce the possibility of search errors.
      correctAnswer <- (for {
        answer <- answerSelector.selectAnswer(denotation, ex.ex.answerOptions)
        _ <- Pnp.require(answer.equals(ex.ex.correctAnswer))
      } yield {
        answer
      }).inOneStep()
    } yield {
      correctAnswer
    }
    
    val score = executor.labelToExecutionScore(ex.ex.diagramLabel)
    PnpExample(unconditional, conditional, Env.init, score)
  }
  
  def getModel: PnpModel = {
    // TODO: need to be able to append parameters from each model.
    parser.model
  }
}