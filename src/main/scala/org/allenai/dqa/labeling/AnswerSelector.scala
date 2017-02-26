package org.allenai.dqa.labeling

import org.allenai.pnp.Pnp

class AnswerSelector {
  
  def selectAnswer(denotation: AnyRef, answerOptions: AnswerOptions): Pnp[Int] = {
    if (denotation.isInstanceOf[Part]) {
      val part = denotation.asInstanceOf[Part]
      val index = answerOptions.matchTokens(part.id)
      if (index >= 0) {
        Pnp.value(index)
      } else {
        Pnp.fail
      }
    } else {
      // TODO
      Pnp.fail
    }
  }
}