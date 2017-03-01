package org.allenai.dqa.matching

import org.allenai.dqa.labeling.DiagramLabel
import org.allenai.dqa.labeling.Diagram
import com.google.common.base.Preconditions

/**
 * Example for diagram part matching model. Each example
 * consists of a source diagram whose parts are to be
 * matched with those of the target diagram. The label
 * is the correct matching. 
 */
case class MatchingExample(source: Diagram, target: Diagram,
    label: MatchingLabel) {
  
}

case class MatchingLabel(targetToSourcePartMap: Map[Int, Int]) {
  def getSourcePartInd(targetPartInd: Int): Int = {
    targetToSourcePartMap(targetPartInd)
  }
}

object MatchingExample {
  
  /**
   * Create a matching example from two diagrams by matching 
   * their equivalently-labeled parts. 
   */
  def fromDiagrams(source: Diagram, sourceLabel: DiagramLabel,
      target: Diagram, targetLabel: DiagramLabel): MatchingExample = {
    
    val partMap = for {
      sourcePart <- source.parts
    } yield {
      val sourcePartLabel = sourceLabel.partLabels(sourcePart.ind)
      val targetInd = targetLabel.partLabels.indexOf(sourcePartLabel) 

      Preconditions.checkState(targetInd != -1, "Could not find part label %s in list %s",
          sourcePartLabel, targetLabel.partLabels)

      (targetInd, sourcePart.ind)
    }

    val label = MatchingLabel(partMap.toMap)

    MatchingExample(source, target, label)
  }
}