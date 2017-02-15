package org.allenai.dqa.labeling

import scala.collection.JavaConverters._

import org.allenai.pnp.Pp
import org.allenai.pnp.PpUtil

import com.google.common.collect.Multimap
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.util.IndexedList
import org.allenai.pnp.ExecutionScore
import org.allenai.pnp.semparse.SemanticParserState
import org.allenai.pnp.Env

/**
 * Executes logical forms against a diagram to produce a
 * denotation, i.e., the answer to the question.
 */
class LabelingExecutor(diagramTypes: IndexedList[String], parts: IndexedList[String], 
    typePartMap: Multimap[Int, Int]) {
 
  import LabelingExecutor._

  val bindings: Map[String, AnyRef] = Map(
      // Function bindings
      ("tokensToPartLabel:<s,l>" -> PpUtil.wrap(tokensToPartLabel _)),
      ("partLabelToPart:<l,p>" -> PpUtil.wrap(partLabelToPart _)),

      // TODO
      ("question:s" -> "TODO: put in the question text.")
      )

  def tokensToPartLabel(tokens: String): Pp[String] = {
    for {
      labeledDiagram <- Pp.getVar[DiagramLabel](DIAGRAM_LABEL_VAR)
      // TODO: use text co-occurrence or something.
      label <- Pp.choose(labeledDiagram.partLabels)
    } yield {
      label
    }
  }

  def partLabelToPart(partLabel: String): Pp[Part] = {
    for {
      labeledDiagram <- Pp.getVar[DiagramLabel](DIAGRAM_LABEL_VAR)
      diagram <- Pp.getVar[Diagram](DIAGRAM_VAR)
      index = labeledDiagram.partLabels.indexOf(partLabel)

      // Fail if we haven't labeled any part with this label.
      _ <- Pp.require(index >= 0)
    } yield {
      diagram.parts(index)
    }
  }
  
  /** Label a diagram with its type and a label for each part.
    */
  def labelDiagram(diagram: Diagram): Pp[DiagramLabel] = {
    for {
      // TODO: parameters
      diagramType <- Pp.chooseTag(diagramTypes.items().asScala, diagram)
      // TODO: don't treat parts as independent.
      permittedLabels = typePartMap.get(diagramTypes.getIndex(diagramType))
        .asScala.map(parts.get(_)).toList
      
      partLabels <- PpUtil.map((x: Part) => labelPart(x, permittedLabels), diagram.parts.toList) 
    } yield {
      DiagramLabel(diagramType, partLabels.toVector) 
    }
  }

  /** Label a single diagram part with one of the permitted labels. 
    */
  def labelPart(part: Part, permittedLabels: List[String]): Pp[String] = {
    for {
      // TODO: parameters
      partLabel <- Pp.chooseTag(permittedLabels, part)
    } yield {
      partLabel
    }
  }

  /**
   * Execute {@code lf} against {@code diagram}.
   */
  def execute(lf: Expression2, diagram: Diagram): Pp[AnyRef] = {
    println("executing: " + lf)

    for {
      // Generate a distribution over labeled diagrams.
      _ <- Pp.setVar(DIAGRAM_VAR, diagram)
      labeledDiagram <- labelDiagram(diagram)
      _ <- Pp.setVar(DIAGRAM_LABEL_VAR, labeledDiagram)
      // Execute the logical form against the labeled diagrams.
      value <- PpUtil.lfToPp(lf, bindings)
    } yield {
      value
    }
  }
  
  /**
   * Get an execution score that assigns 0 probability to
   * executions whose diagram interpretations are inconsistent
   * with {@code label}.
   */
  def labelToExecutionScore(label: DiagramLabel): ExecutionScore = {
    new DiagramLabelExecutionScore(label)
  }
}

class DiagramLabelExecutionScore(val label: DiagramLabel) extends ExecutionScore {
  def apply(tag: Any, choice: Any, env: Env): Double = {
    if (tag != null) {
      if (tag.isInstanceOf[Diagram]) {
        if (choice.equals(label.diagramType)) {
          0.0
        } else {
          Double.NegativeInfinity
        }
      } else if (tag.isInstanceOf[Part]) {
        val part = tag.asInstanceOf[Part]
        if (label.partLabels.length > part.ind && choice.equals(label.partLabels(part.ind))) {
          0.0
        } else {
          Double.NegativeInfinity
        }
      } else {
        // Unknown tag type
        0.0
      }
    } else {
      0.0
    }
  }
}

object LabelingExecutor {
    
  type PartLabel = String

  val DIAGRAM_VAR = "diagram"
  val DIAGRAM_LABEL_VAR = "diagramLabel"
}
