package org.allenai.dqa.matching

import org.allenai.dqa.labeling.Diagram
import org.allenai.dqa.labeling.Part
import org.allenai.dqa.labeling.PointExpressions
import org.allenai.pnp.CompGraph
import org.allenai.pnp.Env
import org.allenai.pnp.ExecutionScore
import org.allenai.pnp.Pnp
import org.allenai.pnp.Pnp.computationGraph
import org.allenai.pnp.PnpModel

import com.google.common.base.Preconditions

import edu.cmu.dynet._
import edu.cmu.dynet.DyNetScalaHelpers._
import edu.cmu.dynet.dynet_swig._

/**
 * Structured prediction model for diagram part matching.
 * Given two diagrams with parts, this model predicts a
 * distribution over matchings between the two sets of
 * parts. 
 */
class MatchingModel(var matchIndependent: Boolean,
    val globalNn: Boolean, val model: PnpModel) {

  import MatchingModel._
  
  def apply(source: Diagram, target: Diagram): Pnp[MatchingLabel] = {
    val sourceParts = source.parts
    val targetParts = target.parts

    for {
      cg <- computationGraph()
      preprocessing = preprocess(source, target, cg)
      matching <- matchRemaining(targetParts.toList, sourceParts.toSet, List(), preprocessing)
    } yield {
      MatchingLabel(matching.map(x => (x._1.ind, x._2.ind)).toMap)
    }
  }

  def preprocess(source: Diagram, target: Diagram, computationGraph: CompGraph): MatchingPreprocessing = {
    val cg = computationGraph.cg
    val sourceFeatures = source.features.getFeatureMatrix(source.parts, cg)
    val targetFeatures = target.features.getFeatureMatrix(target.parts, cg)
    
    val distanceWeights = parameter(cg, computationGraph.getParameter(DISTANCE_WEIGHTS))
    val matchingW1 = parameter(cg, computationGraph.getParameter(MATCHING_W1))
    val matchingB1 = parameter(cg, computationGraph.getParameter(MATCHING_B1))
    val matchingW2 = parameter(cg, computationGraph.getParameter(MATCHING_W2))
    val matchingL = parameter(cg, computationGraph.getParameter(MATCHING_L))
    
    val matchScores = for {
      sourceFeature <- sourceFeatures
    } yield {
      for {
        targetFeature <- targetFeatures
      } yield {
        scoreSourceTargetMatch(sourceFeature, targetFeature, distanceWeights,
            matchingW1, matchingB1, matchingW2, matchingL, computationGraph)
      }
    }
    
    new MatchingPreprocessing(sourceFeatures, targetFeatures, matchScores)
  }

  private def scoreSourceTargetMatch(sourceFeature: PointExpressions,
      targetFeature: PointExpressions,
      distanceWeights: Expression, 
      matchingW1: Expression, matchingB1: Expression, matchingW2: Expression, matchingL: Expression,
      computationGraph: CompGraph): Expression = {
    // Learn a distance metric
    // val delta = sourceFeature.xy - targetFeature.xy
    // val dist = -1.0 * (transpose(delta) * distanceWeights * delta)
    // dist

    // Matching MLP
    val matchingDelta = sourceFeature.matching - targetFeature.matching
    val matchingAbsDelta = rectify(matchingDelta) + rectify(-1 * matchingDelta)
    val matchingAbsScore = matchingW2 * rectify((matchingW1 * matchingAbsDelta) + matchingB1)
    matchingAbsScore

    // Matching linear model
    // val matchingDelta = sourceFeature.matching - targetFeature.matching
    // val matchingAbsDelta = rectify(matchingDelta) + rectify(-1 * matchingDelta)
    // val matchingLinearScore = matchingL * matchingAbsDelta
    // matchingLinearScore

    // No score.
    // input(computationGraph.cg, 0.0f)
  }
  
  /**
   * Gets pairs of vectors encoding the relationship between pairs of
   * parts on the source and target side of the matching. For example,
   * if the matching is [(s1 -> t1), (s2 -> t2)], this generates
   * (r(s1, s2), r(t1, t2)). 
   */
  private def getRelationVectors(matching: List[(Part, Part)],
      preprocessing: MatchingPreprocessing): List[(Expression, Expression)] = {
    for {
      (t1, s1) <- matching
      (t2, s2) <- matching if t2 != t1
    } yield {
      val t1ToT2 = preprocessing.targetFeatures(t2.ind).xy - preprocessing.targetFeatures(t1.ind).xy
      val s1ToS2 = preprocessing.sourceFeatures(s2.ind).xy - preprocessing.sourceFeatures(s1.ind).xy
      (t1ToT2, s1ToS2)
    }
  }
  
  private def getAffineGlobalScore(matching: List[(Part, Part)],
    compGraph: CompGraph, preprocessing: MatchingPreprocessing): Expression = {
    val sourceTargetExpressions = getRelationVectors(matching, preprocessing)
    if (sourceTargetExpressions.size > 2) {
      // Fit a linear regression that transforms the vectors between
      // aligned source points to those of aligned target points.
      val targetMatrix = transpose(concatenate_cols(new ExpressionVector(sourceTargetExpressions.map(_._1))))
      val sourceMatrix = transpose(concatenate_cols(new ExpressionVector(sourceTargetExpressions.map(_._2))))
          
      val identity = input(compGraph.cg, Seq(2, 2), new FloatVector(Vector(1.0f, 0.0f, 0.0f, 1.0f)))
      val l2Regularization = 0.1f
      val regressionParams = inverse((transpose(sourceMatrix) * sourceMatrix)
        + (identity * l2Regularization)) * (transpose(sourceMatrix) * targetMatrix)

      val errors = square(targetMatrix - (sourceMatrix * regressionParams))
      sum_rows(sqrt(sum_rows(errors)))
    } else {
      input(compGraph.cg, 0.0f)
    }
  }

  private def getNnGlobalScore(matching: List[(Part, Part)], 
      compGraph: CompGraph, preprocessing: MatchingPreprocessing,
      transformW2: Expression, deltaW1: Expression, deltaB1: Expression,
      deltaW2: Expression, deltaB2: Expression): Expression = {
    val sourceTargetExpressions = getRelationVectors(matching, preprocessing)
    val concatenatedSourceTargets = sourceTargetExpressions.map(
        x => concatenate(new ExpressionVector(List(x._1, x._2))))

    // Apply a two-layer MLP to each input
    val transformed = concatenatedSourceTargets.map(x => 
      rectify(deltaW2 * rectify((deltaW1 * x) + deltaB1) + deltaB2)
      )

    if (transformed.length > 0) {
      transformW2 * sum(new ExpressionVector(transformed))
    } else {
      input(compGraph.cg, 0.0f)
    }
  }

  /**
   * Get a score for matching targetPart against each source part
   * in remainingArray, given the currentMatching. 
   */
  private def getScores(targetPart: Part, remainingArray: Array[Part],
      currentMatching: List[(Part, Part)], compGraph: CompGraph,
      preprocessing: MatchingPreprocessing): Expression = {
    val unaryScores = remainingArray.map(x => preprocessing.getMatchScore(x, targetPart))

    /*
    val globalScores = if (binaryFactors) {
      val currentMse = getAffineGlobalScore(currentMatching, compGraph, preprocessing)
      val affineTransformParam = parameter(compGraph.cg, compGraph.getParameter(AFFINE_TRANSFORM_PARAM))
      
      remainingArray.map { curSource => 
        val candidateMatching = (targetPart, curSource) :: currentMatching
        val candidateMse = getAffineGlobalScore(candidateMatching, compGraph, preprocessing)
        affineTransformParam * (candidateMse - currentMse)
      }
    } else {
      remainingArray.map(x => input(compGraph.cg, 0.0f))
    }
    */

    val globalScores = if (globalNn) {
      val transformW2 = parameter(compGraph.cg, compGraph.getParameter(TRANSFORM_W1))
      val deltaW1 = parameter(compGraph.cg, compGraph.getParameter(DELTA_W1))
      val deltaB1 = parameter(compGraph.cg, compGraph.getParameter(DELTA_B1))
      val deltaW2 = parameter(compGraph.cg, compGraph.getParameter(DELTA_W2))
      val deltaB2 = parameter(compGraph.cg, compGraph.getParameter(DELTA_B2))
      val currentNnScore = getNnGlobalScore(currentMatching, compGraph, preprocessing,
          transformW2, deltaW1, deltaB1, deltaW2, deltaB2)
      
      remainingArray.map { curSource => 
        val candidateMatching = (targetPart, curSource) :: currentMatching
        val candidateNnScore = getNnGlobalScore(candidateMatching, compGraph, preprocessing,
            transformW2, deltaW1, deltaB1, deltaW2, deltaB2)

        candidateNnScore - currentNnScore
      }
    } else {
      remainingArray.map(x => input(compGraph.cg, 0.0f))
    }

    val scores = unaryScores.zip(globalScores).map(x => x._1 + x._2)

    concatenate(new ExpressionVector(scores.toVector))
  }

  /**
   * Match each part in targetParts with a distinct part in
   * remainingSourceParts. 
   */
  private def matchRemaining(targetParts: List[Part], remainingSourceParts: Set[Part],
      previousMatching: List[(Part, Part)], preprocessing: MatchingPreprocessing): Pnp[List[(Part, Part)]] = {
    if (targetParts.length == 0) {
      Pnp.value(previousMatching)
    } else {
      val targetPart = targetParts.head
      val remainingArray = remainingSourceParts.toArray

      for {
        cg <- computationGraph()
        scoresExpression = getScores(targetPart, remainingArray, previousMatching,
            cg, preprocessing)
        chosenSourcePart <- Pnp.choose(remainingArray, scoresExpression, targetPart)
        nextSourceParts = if (matchIndependent) {
          remainingSourceParts
        } else {
          remainingSourceParts - chosenSourcePart
        }
        matching = (targetPart, chosenSourcePart) :: previousMatching
        
        rest <- matchRemaining(targetParts.tail, nextSourceParts, matching, preprocessing)
      } yield {
        rest
      }
    }
  }

  def getLabelOracle(label: MatchingLabel): MatchingExecutionScore = {
    MatchingExecutionScore(label)
  }
  
  def save(saver: ModelSaver): Unit = {
    saver.add_boolean(matchIndependent)
    saver.add_boolean(globalNn)
  }
}

/**
 * Stores the output of preprocessing and neural network
 * computations that can be shared across many choices.
 */
class MatchingPreprocessing(val sourceFeatures: Array[PointExpressions],
    val targetFeatures: Array[PointExpressions], val matchScores: Array[Array[Expression]]) {
  
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

  val DISTANCE_WEIGHTS = "distanceWeights"
  
  val MATCHING_W1 = "matchingW1"
  val MATCHING_B1 = "matchingB1"
  val MATCHING_W2 = "matchingW2"
  val MATCHING_L = "matchingL"

  val AFFINE_TRANSFORM_PARAM = "affineTransformParam"
  
  val TRANSFORM_W1 = "transformW1"
  val DELTA_W1 = "deltaW1"
  val DELTA_B1 = "deltaB1"
  val DELTA_W2 = "deltaW2"
  val DELTA_B2 = "deltaB2"

  /**
   * Create a MatchingModel and populate {@code model} with the
   * necessary neural network parameters.
   */
  def create(xyFeatureDim: Int, matchingFeatureDim: Int,
      vggFeatureDim: Int, matchIndependent: Boolean,
      globalNn: Boolean, model: PnpModel): MatchingModel = {
    model.addParameter(DISTANCE_WEIGHTS, Seq(xyFeatureDim, xyFeatureDim))

    val matchingHiddenDim = 768
    model.addParameter(MATCHING_W1, Seq(matchingHiddenDim, matchingFeatureDim))
    model.addParameter(MATCHING_B1, Seq(matchingHiddenDim))
    model.addParameter(MATCHING_W2, Seq(1, matchingHiddenDim))
    model.addParameter(MATCHING_L, Seq(1, matchingFeatureDim))
    
    model.addParameter(AFFINE_TRANSFORM_PARAM, Seq(1))
    
    val transformHiddenDim1 = 32
    val transformHiddenDim2 = 32
    val deltaDim = 4
    model.addParameter(TRANSFORM_W1, Seq(1, transformHiddenDim2))
    model.addParameter(DELTA_W1, Seq(transformHiddenDim1, deltaDim))
    model.addParameter(DELTA_B1, Seq(transformHiddenDim1))
    model.addParameter(DELTA_W2, Seq(transformHiddenDim2, transformHiddenDim1))
    model.addParameter(DELTA_B2, Seq(transformHiddenDim2))

    new MatchingModel(matchIndependent, globalNn, model)
  }

  /**
   * Load a serialized MatchingModel.
   */
  def load(loader: ModelLoader, model: PnpModel): MatchingModel = {
    val matchIndependent = loader.load_boolean()
    val globalNn = loader.load_boolean()
    new MatchingModel(matchIndependent, globalNn, model)
  }
}