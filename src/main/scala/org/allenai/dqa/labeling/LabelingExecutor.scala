package org.allenai.dqa.labeling

import scala.collection.JavaConverters._

import org.allenai.pnp.CompGraph
import org.allenai.pnp.Env
import org.allenai.pnp.ExecutionScore.ExecutionScore
import org.allenai.pnp.Pnp
import org.allenai.pnp.PnpUtil

import com.google.common.collect.HashMultimap
import com.google.common.collect.Multimap
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.dynet._
import edu.cmu.dynet.DyNetScalaHelpers._
import edu.cmu.dynet.dynet_swig._
import org.allenai.pnp.PnpModel

/**
 * Executes logical forms against a diagram to produce a
 * denotation, i.e., the answer to the question.
 */
class LabelingExecutor(diagramTypes: IndexedList[String], parts: IndexedList[String], 
    typePartMap: Multimap[Int, Int], partFeatureDim: Int) {
 
  import LabelingExecutor._

  val bindings: Map[String, AnyRef] = Map(
      // Function bindings
      ("tokensToPartLabel:<s,l>" -> PnpUtil.wrap(tokensToPartLabel _)),
      ("partLabelToPart:<l,p>" -> PnpUtil.wrap(partLabelToPart _)),

      // TODO
      ("question:s" -> "TODO: put in the question text.")
      )

  def tokensToPartLabel(tokens: String): Pnp[String] = {
    for {
      labeledDiagram <- Pnp.getVar[DiagramLabel](DIAGRAM_LABEL_VAR)
      biasParam <- Pnp.param(DIAGRAM_PART_BIAS_PREFIX + labeledDiagram.diagramType)
      
      // TODO: use text co-occurrence or something.
      label <- Pnp.choose(labeledDiagram.partLabels.toArray, biasParam)
    } yield {
      label
    }
  }

  def partLabelToPart(partLabel: String): Pnp[Part] = {
    for {
      labeledDiagram <- Pnp.getVar[DiagramLabel](DIAGRAM_LABEL_VAR)
      diagram <- Pnp.getVar[Diagram](DIAGRAM_VAR)
      index = labeledDiagram.partLabels.indexOf(partLabel)

      // Fail if we haven't labeled any part with this label.
      _ <- Pnp.require(index >= 0)
    } yield {
      diagram.parts(index)
    }
  }
  
  /** Label a diagram with its type and a label for each part.
    */
  def labelDiagram(diagram: Diagram): Pnp[DiagramLabel] = {
    for {
      // TODO: parameters
      diagramType <- Pnp.chooseTag(diagramTypes.items().asScala, diagram)

      permittedLabels = typePartMap.get(diagramTypes.getIndex(diagramType))
        .asScala.map(parts.get(_)).toArray

      cg <- Pnp.computationGraph()
      // TODO: don't treat parts as independent.        
      partLabels <- PnpUtil.map((x: Part) => labelPart(x, permittedLabels, cg, diagram),
          diagram.parts.toList) 
    } yield {
      DiagramLabel(diagramType, partLabels.toVector) 
    }
  }

  /** Label a single diagram part with one of the permitted labels. 
    */
  def labelPart(part: Part, permittedLabels: Array[String], cg: CompGraph,
      diagram: Diagram): Pnp[String] = {
    val scores = for {
      label <- permittedLabels
    } yield {
      val params = parameter(cg.cg, cg.getParameter(PART_PREFIX + label))
      val featureVector = input(cg.cg, Seq(partFeatureDim),
          diagram.features.getFeatures(part).xy)
      dot_product(params, featureVector)
    }

    val scoreExpression = concatenate(new ExpressionVector(scores.toList))

    for {
      partLabel <- Pnp.choose(permittedLabels, scoreExpression, part)
    } yield {
      partLabel
    }
  }
  
  /**
   * Execute {@code lf} against {@code diagram}.
   */
  def execute(lf: Expression2, diagram: Diagram): Pnp[AnyRef] = {
    println("executing: " + lf)

    for {
      // Generate a distribution over labeled diagrams.
      _ <- Pnp.setVar(DIAGRAM_VAR, diagram)
      labeledDiagram <- labelDiagram(diagram)
      _ <- Pnp.setVar(DIAGRAM_LABEL_VAR, labeledDiagram)
      // Execute the logical form against the labeled diagrams.
      value <- PnpUtil.lfToPnp(lf, bindings)
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
  
  val PART_PREFIX = "part:"
  val DIAGRAM_PART_BIAS_PREFIX = "diagramPartBias:"
  
  def create(diagramTypes: IndexedList[String], parts: IndexedList[String], 
      typePartMap: Multimap[Int, Int], partFeatureDim: Int, model: PnpModel): LabelingExecutor = {
    for (part <- parts.items().asScala) {
      model.addParameter(PART_PREFIX + part, Seq(partFeatureDim))
    }
    for (diagram <- diagramTypes.items().asScala) {
      val index = diagramTypes.getIndex(diagram)
      val numParts = typePartMap.get(index).size()
      model.addParameter(DIAGRAM_PART_BIAS_PREFIX + diagram, Seq(numParts))
    }
    
    new LabelingExecutor(diagramTypes, parts, typePartMap, partFeatureDim)
  }
  
  /**
   * Create a labeling executor whose diagram type and
   * part vocabulary is constructed from diagramLabels.
   */
  def fromLabels(diagramLabels: Array[DiagramLabel], partFeatureDim: Int,
      model: PnpModel): LabelingExecutor = {
    val diagramTypes = IndexedList.create[String]
    val diagramParts = IndexedList.create[String]
    val typePartMap = HashMultimap.create[Int, Int]
    for (label <- diagramLabels) {
      val diagramTypeId = diagramTypes.add(label.diagramType)
      for (part <- label.partLabels) {
        val diagramPartId = diagramParts.add(part)
        typePartMap.put(diagramTypeId, diagramPartId)
      }
    }

    create(diagramTypes, diagramParts, typePartMap, partFeatureDim, model)
  }
}
