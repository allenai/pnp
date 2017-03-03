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
import org.allenai.pnp.CompGraph

/**
 * Structured prediction model for diagram part matching.
 * Given two diagrams with parts, this model predicts a
 * distribution over matchings between the two sets of
 * parts. 
 */
class MatchingModel(val featureDim: Int, val matchIndependent: Boolean,
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
    
    val matchScores = for {
      sourceFeature <- sourceFeatures
    } yield {
      for {
        targetFeature <- targetFeatures
      } yield {
        // Learn a distance metric
        val delta = sourceFeature - targetFeature
        val dist = -1.0 * (transpose(delta) * distanceWeights * delta)
        
        // multilayer perceptron
        val concatenated = concatenate(new ExpressionVector(List(sourceFeature, targetFeature)))
        val mlpOutput = (mlpL2W * tanh((mlpL1W * concatenated) + mlpL1B)) 

        // mlpOutput
        dist
        // -1.0 * dot_product(delta, delta)
        // input(cg, 0.0f)
      }
    }

    new MatchingPreprocessing(sourceFeatures, targetFeatures, matchScores)
  }

  private def getScores(targetPart: Part, remainingArray: Array[Part],
      currentMatching: List[(Part, Part)], compGraph: CompGraph,
      preprocessing: MatchingPreprocessing): Expression = {
    val cg = compGraph.cg
    val unaryScores = remainingArray.map(x => preprocessing.getMatchScore(x, targetPart))
    
    val scores = if (binaryFactors) {
      val binaryDistanceWeights = parameter(cg,
          compGraph.getParameter(BINARY_DISTANCE_WEIGHTS))
      val binaryW = parameter(cg, compGraph.getParameter(BINARY_W))
      val binaryB = parameter(cg, compGraph.getParameter(BINARY_B))
      val binaryDistanceWeightsNonlinear = parameter(cg,
          compGraph.getParameter(BINARY_HIDDEN_DIST))
          
      val curTargetFeatures = preprocessing.targetFeatures(targetPart.ind)
      
      val binaryScores = for {
        curSource <- remainingArray
      } yield {
        val curSourceFeatures = preprocessing.sourceFeatures(curSource.ind)
        
        val pairwiseScores = for {
          (prevTarget, prevSource) <- currentMatching 
        } yield {
          val prevTargetFeatures = preprocessing.targetFeatures(prevTarget.ind)
          val prevSourceFeatures = preprocessing.sourceFeatures(prevSource.ind)
          
          val prevToCurTarget = curTargetFeatures - prevTargetFeatures
          val prevToCurSource = curSourceFeatures - prevSourceFeatures

          val prevToCurTargetNonlinear = tanh((binaryW * prevToCurTarget) + binaryB)
          val prevToCurSourceNonlinear = tanh((binaryW * prevToCurSource) + binaryB)

          /*
          val delta = prevToCurTargetNonlinear - prevToCurSourceNonlinear
          val dist = -1.0 * (transpose(delta) * binaryDistanceWeightsNonlinear * delta)
          dist
          */

          // learn a distance metric (assuming binaryDistanceWeights is PSD)
          val delta = prevToCurTarget - prevToCurSource
          val dist = -1.0 * (transpose(delta) * binaryDistanceWeights * delta)
          dist
          
          // val sim = dot_product(prevToCurTarget, prevToCurSource)
          
          // transpose(prevToCurTarget) * binaryDistanceWeights * prevToCurSource

          /*
          val delta = prevToCurTarget - prevToCurSource
          val dist = -1.0 * dot_product(delta, delta)
          dist
          */
        }

        pairwiseScores.foldLeft(input(cg, 0))(_ + _)
      }

      unaryScores.zip(binaryScores).map(x => x._1 + x._2)
    } else {
      unaryScores
    }

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
    saver.add_int(featureDim)
    saver.add_boolean(matchIndependent)
    saver.add_boolean(binaryFactors)
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

  val DISTANCE_WEIGHTS = "sourceTargetWeights"
  val BINARY_DISTANCE_WEIGHTS = "binaryDistanceWeights"
  val BINARY_W = "binaryW"
  val BINARY_B = "binaryB"
  val BINARY_HIDDEN_DIST = "binaryHiddenWeights"

  val MLP_L1_W = "mlpL1W"
  val MLP_L1_B = "mlpL1B"
  val MLP_L2_W = "mlpL2W"
  
  /**
   * Create a MatchingModel and populate {@code model} with the
   * necessary neural network parameters.
   */
  def create(featureDim: Int, matchIndependent: Boolean,
      binaryFactors: Boolean, model: PnpModel): MatchingModel = {
    val hiddenDim = 10
    model.addParameter(DISTANCE_WEIGHTS, Seq(featureDim, featureDim))
    model.addParameter(BINARY_DISTANCE_WEIGHTS, Seq(featureDim, featureDim))
    model.addParameter(BINARY_W, Seq(hiddenDim, featureDim))
    model.addParameter(BINARY_B, Seq(hiddenDim))
    model.addParameter(BINARY_HIDDEN_DIST, Seq(hiddenDim, hiddenDim))
    model.addParameter(MLP_L1_W, Seq(hiddenDim, featureDim * 2))
    model.addParameter(MLP_L1_B, Seq(hiddenDim))
    model.addParameter(MLP_L2_W, Seq(1, hiddenDim))
    new MatchingModel(featureDim, matchIndependent, binaryFactors, model)
  }

  /**
   * Load a serialized MatchingModel.
   */
  def load(loader: ModelLoader, model: PnpModel): MatchingModel = {
    val featureDim = loader.load_int()
    val matchIndependent = loader.load_boolean()
    val binaryFactors = loader.load_boolean()
    new MatchingModel(featureDim, matchIndependent, binaryFactors, model)
  }
}