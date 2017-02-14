package org.allenai.dqa.labeling

import scala.collection.JavaConverters._

import org.allenai.pnp.Pp
import org.allenai.pnp.PpUtil

import com.google.common.collect.Multimap
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.util.IndexedList

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
      diagramType <- Pp.choose(diagramTypes.items().asScala)
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
      part <- Pp.choose(permittedLabels)
    } yield {
      part
    }
  }
  
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
}

object LabelingExecutor {
    
  type PartLabel = String

  val DIAGRAM_VAR = "diagram"
  val DIAGRAM_LABEL_VAR = "diagramLabel"
}
