package org.allenai.dqa.labeling

import org.allenai.pnp.Pp

class AnswerSelector {
  
  def selectAnswer(denotation: AnyRef, answerOptions: AnswerOptions): Pp[Int] = {
    if (denotation.isInstanceOf[Part]) {
      val part = denotation.asInstanceOf[Part]
      val index = answerOptions.matchTokens(part.id)
      if (index >= 0) {
        Pp.value(index)
      } else {
        Pp.fail
      }
    } else {
      // TODO
      Pp.fail
      // Pp.choose((0 until answerOptions.length).toArray)
    }
  }
}