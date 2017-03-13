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
    val binaryFactors: Boolean, val model: PnpModel) {

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
    val mlpL1W = parameter(cg, computationGraph.getParameter(MLP_L1_W))
    val mlpL1B = parameter(cg, computationGraph.getParameter(MLP_L1_B))
    val mlpL2W = parameter(cg, computationGraph.getParameter(MLP_L2_W))
    val vggW1 = parameter(cg, computationGraph.getParameter(VGG_W1))
    val vggB1 = parameter(cg, computationGraph.getParameter(VGG_B1))
    val vggW2 = parameter(cg, computationGraph.getParameter(VGG_W2))
    val matchingW1 = parameter(cg, computationGraph.getParameter(MATCHING_W1))
    val matchingB1 = parameter(cg, computationGraph.getParameter(MATCHING_B1))
    val matchingW2 = parameter(cg, computationGraph.getParameter(MATCHING_W2))
    val matchingL = parameter(cg, computationGraph.getParameter(MATCHING_L))
    val matchingOuterW1 = parameter(cg, computationGraph.getParameter(MATCHING_OUTER_W1))
    val matchingOuterB1 = parameter(cg, computationGraph.getParameter(MATCHING_OUTER_B1))
    val matchingOuterW2 = parameter(cg, computationGraph.getParameter(MATCHING_OUTER_W2))
    val matchingOuterL = parameter(cg, computationGraph.getParameter(MATCHING_OUTER_L))
    
    val matchScores = for {
      sourceFeature <- sourceFeatures
    } yield {
      for {
        targetFeature <- targetFeatures
      } yield {
        scoreSourceTargetMatch(sourceFeature, targetFeature, distanceWeights,
            mlpL1W, mlpL1B, mlpL2W, vggW1, vggB1, vggW2, matchingW1, matchingB1, matchingW2,
            matchingL, matchingOuterW1, matchingOuterB1, matchingOuterW2, matchingOuterL,
            computationGraph)
      }
    }

    // Create expressions for parameters once up front for efficiency.
    val binaryDistanceWeights = parameter(cg,
        computationGraph.getParameter(BINARY_DISTANCE_WEIGHTS))
    val binaryW = parameter(cg, computationGraph.getParameter(BINARY_W))
    val binaryB = parameter(cg, computationGraph.getParameter(BINARY_B))
    val binaryDistanceWeightsNonlinear = parameter(cg,
        computationGraph.getParameter(BINARY_HIDDEN_DIST))
    
    new MatchingPreprocessing(sourceFeatures, targetFeatures, matchScores,
        binaryDistanceWeights, binaryW, binaryB, binaryDistanceWeightsNonlinear)
  }

  private def scoreSourceTargetMatch(sourceFeature: PointExpressions,
      targetFeature: PointExpressions,
      distanceWeights: Expression, mlpL1W: Expression, mlpL1B: Expression,
      mlpL2W: Expression, vggW1: Expression, vggB1: Expression, vggW2: Expression,
      matchingW1: Expression, matchingB1: Expression, matchingW2: Expression, matchingL: Expression,
      matchingOuterW1: Expression, matchingOuterB1: Expression, matchingOuterW2: Expression,
      matchingOuterL: Expression, computationGraph: CompGraph): Expression = {
    // Learn a distance metric
    // val delta = sourceFeature.xy - targetFeature.xy
    // val dist = -1.0 * (transpose(delta) * distanceWeights * delta)
    // dist

    // val vggDelta = sourceFeature.vggAll - targetFeature.vggAll
    // val vggDist = -1.0 * dot_product(vggDelta, vggDelta)

    // val vggAbsDelta = rectify(vggDelta) + rectify(-1 * vggDelta)
    // val vggDelta = sourceFeature.vgg0 - targetFeature.vgg0
    // val vggAbsScore = vggW2 * rectify((vggW1 * vggDelta) + vggB1)
    // vggAbsScore
    // vggDist

    // matching MLP
    val matchingDelta = sourceFeature.matching - targetFeature.matching
    val matchingAbsDelta = rectify(matchingDelta) + rectify(-1 * matchingDelta)
    val matchingAbsScore = matchingW2 * rectify((matchingW1 * matchingAbsDelta) + matchingB1)
    matchingAbsScore

    // matchingAbsScore + dist

    // Matching linear model
    /*
    val matchingDelta = sourceFeature.matching - targetFeature.matching
    val matchingAbsDelta = rectify(matchingDelta) + rectify(-1 * matchingDelta)
    val matchingLinearScore = matchingL * matchingAbsDelta
    matchingLinearScore
    */

    // Outer product
    /*
    val matchingDim = 32
    val sourceMat = concatenate_cols(new ExpressionVector(Vector.fill(matchingDim)(sourceFeature.matching)))
    val targetMat = transpose(concatenate_cols(new ExpressionVector(Vector.fill(matchingDim)(targetFeature.matching))))
    
    val oprod = reshape(sourceMat * targetMat, Seq(matchingDim * matchingDim))
    // val oprodScore = matchingOuterW2 * oprod
    val oprodScore = matchingOuterW2 * rectify((matchingOuterW1 * oprod) + matchingOuterB1)

    oprodScore
    */
    
    // Outer product of absolute value delta
    /*
    val matchingDim = 32
    val delta = sourceFeature.matching - targetFeature.matching
    val absDelta = rectify(delta) + rectify(-1 * delta)
    val mat = concatenate_cols(new ExpressionVector(Vector.fill(matchingDim)(absDelta)))
    val oprod = reshape(cmult(mat, transpose(mat)), Seq(matchingDim * matchingDim))
    val oprodScore = matchingOuterL * oprod
    oprodScore
    */
    
    // dist
    
    // multilayer perceptron
    // val concatenated = concatenate(new ExpressionVector(List(sourceFeature, targetFeature)))
    // val mlpOutput = (mlpL2W * tanh((mlpL1W * concatenated) + mlpL1B))

    // mlpOutput
    // dist
    // -1.0 * dot_product(delta, delta)
    // input(computationGraph.cg, 0.0f)
  }
  
  private def getAffineTransformData(matching: List[(Part, Part)],
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
  
  private def getAffineTransformSse(matching: List[(Part, Part)],
    compGraph: CompGraph, preprocessing: MatchingPreprocessing): Expression = {
    val sourceTargetExpressions = getAffineTransformData(matching, preprocessing)
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
    
  private def getNnTransformScore(matching: List[(Part, Part)], 
      compGraph: CompGraph, preprocessing: MatchingPreprocessing,
      transformW1: Expression, transformB1: Expression,
      transformW2: Expression, deltaW1: Expression, deltaB1: Expression,
      deltaW2: Expression, deltaB2: Expression): Expression = {
    val sourceTargetExpressions = getAffineTransformData(matching, preprocessing)
    val concatenatedSourceTargets = sourceTargetExpressions.map(
        x => concatenate(new ExpressionVector(List(x._1, x._2))))

    val transformed = concatenatedSourceTargets.map(x => 
      rectify(deltaW2 * rectify((deltaW1 * x) + deltaB1) + deltaB2)
      )
    
    // This might be too complicated?
    // transformW2 * rectify((transformW1 * sum(new ExpressionVector(transformed))) + transformB1)
    // transformW2 * sum(new ExpressionVector(transformed))
    // input(compGraph.cg, 0.0f)

    if (transformed.length > 0) {
      transformW2 * sum(new ExpressionVector(transformed))
    } else {
      input(compGraph.cg, 0.0f)
    }
  }

  private def getScores(targetPart: Part, remainingArray: Array[Part],
      currentMatching: List[(Part, Part)], compGraph: CompGraph,
      preprocessing: MatchingPreprocessing): Expression = {
    val unaryScores = remainingArray.map(x => preprocessing.getMatchScore(x, targetPart))

    /*
    val affineTransformScores = if (binaryFactors) {
      val currentMse = getAffineTransformSse(currentMatching, compGraph, preprocessing)
      val affineTransformParam = parameter(compGraph.cg, compGraph.getParameter(AFFINE_TRANSFORM_PARAM))
      
      remainingArray.map { curSource => 
        val candidateMatching = (targetPart, curSource) :: currentMatching
        val candidateMse = getAffineTransformSse(candidateMatching, compGraph, preprocessing)
        affineTransformParam * (candidateMse - currentMse)
      }
    } else {
      remainingArray.map(x => input(compGraph.cg, 0.0f))
    }
    */

    val affineTransformScores = if (binaryFactors) {
      val transformW1 = parameter(compGraph.cg, compGraph.getParameter(TRANSFORM_W1))
      val transformB1 = parameter(compGraph.cg, compGraph.getParameter(TRANSFORM_B1))
      val transformW2 = parameter(compGraph.cg, compGraph.getParameter(TRANSFORM_W2))
      val deltaW1 = parameter(compGraph.cg, compGraph.getParameter(DELTA_W1))
      val deltaB1 = parameter(compGraph.cg, compGraph.getParameter(DELTA_B1))
      val deltaW2 = parameter(compGraph.cg, compGraph.getParameter(DELTA_W2))
      val deltaB2 = parameter(compGraph.cg, compGraph.getParameter(DELTA_B2))
      val currentNnScore = getNnTransformScore(currentMatching, compGraph, preprocessing,
          transformW1, transformB1, transformW2, deltaW1, deltaB1, deltaW2, deltaB2)
      
      remainingArray.map { curSource => 
        val candidateMatching = (targetPart, curSource) :: currentMatching
        val candidateNnScore = getNnTransformScore(candidateMatching, compGraph, preprocessing, transformW1,
          transformB1, transformW2, deltaW1, deltaB1, deltaW2, deltaB2)

        candidateNnScore - currentNnScore
      }
    } else {
      remainingArray.map(x => input(compGraph.cg, 0.0f))
    }

    val scores = unaryScores.zip(affineTransformScores).map(x => x._1 + x._2)

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
    saver.add_boolean(binaryFactors)
  }
}

/**
 * Stores the output of preprocessing and neural network
 * computations that can be shared across many choices.
 */
class MatchingPreprocessing(val sourceFeatures: Array[PointExpressions],
    val targetFeatures: Array[PointExpressions], val matchScores: Array[Array[Expression]],
    val binaryDistanceWeights: Expression, val binaryW: Expression, val binaryB: Expression,
    val binaryDistanceWeightsNonlinear: Expression) {
  
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

  val DISTANCE_WEIGHTS = "sourceTargetWeights"
  val BINARY_DISTANCE_WEIGHTS = "binaryDistanceWeights"
  val BINARY_W = "binaryW"
  val BINARY_B = "binaryB"
  val BINARY_HIDDEN_DIST = "binaryHiddenWeights"

  val MLP_L1_W = "mlpL1W"
  val MLP_L1_B = "mlpL1B"
  val MLP_L2_W = "mlpL2W"
  
  val VGG_W1 = "vggW1"
  val VGG_B1 = "vggB1"
  val VGG_W2 = "vggW2"
  
  val MATCHING_W1 = "matchingW1"
  val MATCHING_B1 = "matchingB1"
  val MATCHING_W2 = "matchingW2"
  val MATCHING_L = "matchingL"
  
  val MATCHING_OUTER_W1 = "matchingOuterW1"
  val MATCHING_OUTER_B1 = "matchingOuterB1"
  val MATCHING_OUTER_W2 = "matchingOuterW2"
  val MATCHING_OUTER_L = "matchingOuterL"

  val AFFINE_TRANSFORM_PARAM = "AFFINE_TRANSFORM_PARAM"
  
  val TRANSFORM_W1 = "transformW1"
  val TRANSFORM_B1 = "transformB1"
  val TRANSFORM_W2 = "transformW2"
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
      binaryFactors: Boolean, model: PnpModel): MatchingModel = {
    val hiddenDim = 10
    model.addParameter(DISTANCE_WEIGHTS, Seq(xyFeatureDim, xyFeatureDim))
    model.addParameter(BINARY_DISTANCE_WEIGHTS, Seq(xyFeatureDim, xyFeatureDim))
    model.addParameter(BINARY_W, Seq(hiddenDim, xyFeatureDim))
    model.addParameter(BINARY_B, Seq(hiddenDim))
    model.addParameter(BINARY_HIDDEN_DIST, Seq(hiddenDim, hiddenDim))
    model.addParameter(MLP_L1_W, Seq(hiddenDim, xyFeatureDim * 2))
    model.addParameter(MLP_L1_B, Seq(hiddenDim))
    model.addParameter(MLP_L2_W, Seq(1, hiddenDim))
    
    // TODO: 512
    val vggHiddenDim = 32
    model.addParameter(VGG_W1, Seq(vggHiddenDim, vggFeatureDim))
    model.addParameter(VGG_B1, Seq(vggHiddenDim))
    model.addParameter(VGG_W2, Seq(1, vggHiddenDim))
    
    val matchingHiddenDim = 768
    model.addParameter(MATCHING_W1, Seq(matchingHiddenDim, matchingFeatureDim))
    model.addParameter(MATCHING_B1, Seq(matchingHiddenDim))
    model.addParameter(MATCHING_W2, Seq(1, matchingHiddenDim))
    model.addParameter(MATCHING_L, Seq(1, matchingFeatureDim))
    
    val matchingOuterDim = matchingFeatureDim * matchingFeatureDim
    model.addParameter(MATCHING_OUTER_W1, Seq(matchingHiddenDim, matchingOuterDim))
    model.addParameter(MATCHING_OUTER_B1, Seq(matchingHiddenDim))
    model.addParameter(MATCHING_OUTER_W2, Seq(1, matchingHiddenDim))
    model.addParameter(MATCHING_OUTER_L, Seq(1, matchingOuterDim))
    
    model.addParameter(AFFINE_TRANSFORM_PARAM, Seq(1))
    
    val transformHiddenDim1 = 32
    val transformHiddenDim2 = 32
    val deltaDim = 4
    model.addParameter(TRANSFORM_W1, Seq(transformHiddenDim2, transformHiddenDim2))
    model.addParameter(TRANSFORM_B1, Seq(transformHiddenDim2))
    model.addParameter(TRANSFORM_W2, Seq(1, transformHiddenDim2))
    model.addParameter(DELTA_W1, Seq(transformHiddenDim1, deltaDim))
    model.addParameter(DELTA_B1, Seq(transformHiddenDim1))
    model.addParameter(DELTA_W2, Seq(transformHiddenDim2, transformHiddenDim1))
    model.addParameter(DELTA_B2, Seq(transformHiddenDim2))


    new MatchingModel(matchIndependent, binaryFactors, model)
  }

  /**
   * Load a serialized MatchingModel.
   */
  def load(loader: ModelLoader, model: PnpModel): MatchingModel = {
    val matchIndependent = loader.load_boolean()
    val binaryFactors = loader.load_boolean()
    new MatchingModel(matchIndependent, binaryFactors, model)
  }
}