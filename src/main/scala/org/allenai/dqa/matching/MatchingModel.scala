package org.allenai.dqa.matching

import org.allenai.dqa.labeling.Diagram
import org.allenai.dqa.labeling.Part
import org.allenai.pnp.Pnp
import org.allenai.pnp.Pnp.computationGraph
import org.allenai.pnp.PnpModel

import edu.cmu.dynet._
import edu.cmu.dynet.DyNetScalaHelpers._
import edu.cmu.dynet.dynet_swig._
import org.allenai.pnp.ExecutionScore
import com.google.common.base.Preconditions
import org.allenai.pnp.Env

/**
 * Structured prediction model for diagram part matching.
 * Given two diagrams with parts, this model predicts a
 * distribution over matchings between the two sets of
 * parts. 
 */
class MatchingModel(val featureDim: Int, val sourceTargetParam: Parameter,
    val model: PnpModel) {
  
  def apply(source: Diagram, target: Diagram): Pnp[MatchingLabel] = {
    val sourceParts = source.parts
    val targetParts = target.parts

    for {
      cg <- computationGraph()
      preprocessing = preprocess(source, target, cg.cg)
      matching <- matchRemaining(targetParts.toList, sourceParts.toSet, preprocessing)
    } yield {
      MatchingLabel(matching.map(x => (x._1.ind, x._2.ind)).toMap)
    }
  }

  def preprocess(source: Diagram, target: Diagram, cg: ComputationGraph): MatchingPreprocessing = {
    val sourceFeatures = source.features.getFeatureMatrix(source.parts, cg)
    val targetFeatures = target.features.getFeatureMatrix(target.parts, cg)

    val weights = parameter(cg, sourceTargetParam)
    
    val matchScores = for {
      sourceFeature <- sourceFeatures
    } yield {
      for {
        targetFeature <- targetFeatures
      } yield {
        // TODO: different scoring function
        transpose(sourceFeature) * weights * targetFeature
      }
    }

    new MatchingPreprocessing(sourceFeatures, targetFeatures, matchScores)
  }

  /**
   * Match each part in targetParts with a distinct part in
   * remainingSourceParts. 
   */
  private def matchRemaining(targetParts: List[Part], remainingSourceParts: Set[Part],
      preprocessing: MatchingPreprocessing): Pnp[List[(Part, Part)]] = {
    if (targetParts.length == 0) {
      Pnp.value(List())
    } else {
      val targetPart = targetParts.head
      val remainingArray = remainingSourceParts.toArray
      val scores = remainingArray.map(x => preprocessing.getMatchScore(x, targetPart))
      val scoresExpression = concatenate(new ExpressionVector(scores.toVector))
      
      for {
        chosenSourcePart <- Pnp.choose(remainingArray, scoresExpression, targetPart)
        rest <- matchRemaining(targetParts.tail, remainingSourceParts - chosenSourcePart, preprocessing)
      } yield {
        (targetPart, chosenSourcePart) :: rest
      }
    }
  }

  def getLabelOracle(label: MatchingLabel): MatchingExecutionScore = {
    MatchingExecutionScore(label)
  }
  
  def save(saver: ModelSaver): Unit = {
    // TODO
    saver.add_int(featureDim)
  }
}

/**
 * Stores the output of preprocessing and neural network
 * computations that can be shared across many choices.
 */
class MatchingPreprocessing(val sourceFeatures: Array[Expression],
    val targetFeatures: Array[Expression], val matchScores: Array[Array[Expression]]) {
  
  
  def getMatchScore(sourcePart: Part, targetPart: Part): Expression = {
    matchScores(sourcePart.ind)(targetPart.ind)
  }
}

case class MatchingExecutionScore(label: MatchingLabel) extends ExecutionScore {
  def apply(tag: Any, choice: Any, env: Env): Double = {
    if (tag != null && tag.isInstanceOf[Part]) {
      val targetPart = tag.asInstanceOf[Part]
      
      Preconditions.checkArgument(choice.isInstanceOf[Part])
      val sourcePart = choice.asInstanceOf[Part]

      if (label.getSourcePartInd(targetPart.ind) == sourcePart.ind) {
        0.0
      } else {
        Double.NegativeInfinity
      }
    } else {
      0.0
    }
  }
}

object MatchingModel {

  val SOURCE_TARGET_WEIGHTS = "sourceTargetWeights"
  
  /**
   * Create a MatchingModel and populate {@code model} with the
   * necessary neural network parameters.
   */
  def create(featureDim: Int, model: PnpModel): MatchingModel = {
    val sourceTargetWeights = model.addParameter(SOURCE_TARGET_WEIGHTS, Seq(featureDim, featureDim))
    new MatchingModel(featureDim, sourceTargetWeights, model)
  }

  /**
   * Load a serialized MatchingModel.
   */
  def load(loader: ModelLoader, model: PnpModel): MatchingModel = {
    val featureDim = loader.load_int()
    new MatchingModel(featureDim, model.getParameter(SOURCE_TARGET_WEIGHTS), model)
  }
}