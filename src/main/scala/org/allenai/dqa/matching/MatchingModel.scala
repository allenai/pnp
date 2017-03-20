package org.allenai.dqa.matching

import scala.Vector
import scala.collection.JavaConverters._

import org.allenai.dqa.labeling.Diagram
import org.allenai.dqa.labeling.DiagramLabel
import org.allenai.dqa.labeling.Part
import org.allenai.dqa.labeling.PointExpressions
import org.allenai.pnp.CompGraph
import org.allenai.pnp.Env
import org.allenai.pnp.ExecutionScore.ExecutionScore
import org.allenai.pnp.Pnp
import org.allenai.pnp.Pnp.computationGraph
import org.allenai.pnp.PnpModel

import com.google.common.base.Preconditions
import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.dynet._
import scala.util.Random
import org.allenai.dqa.labeling.DiagramFeatures
import Expression._

/**
 * Structured prediction model for diagram part matching.
 * Given two diagrams with parts, this model predicts a
 * distribution over matchings between the two sets of
 * parts. 
 */
class MatchingModel(var config: MatchingModelConfig,
    val forwardBuilder: LstmBuilder, val backwardBuilder: LstmBuilder,
    val contextBuilder: LstmBuilder,
    val pointerNetInputBuilder: LstmBuilder, val pointerNetOutputBuilder: LstmBuilder,
    val labelDict: IndexedList[String], val model: PnpModel) {

  import MatchingModel._
  
  def apply(source: Diagram, sourceLabel: DiagramLabel, target: Diagram): Pnp[MatchingLabel] = {
    val sourceParts = source.parts
    val targetParts = Random.shuffle(target.parts)

    for {
      cg <- computationGraph()
      preprocessing = preprocess(source, sourceLabel, target, targetParts, cg)
      // pointerNetInitialInput = parameter(cg.cg, cg.getParameter(POINTER_NET_INIT))
      
      matching <- matchRemaining(targetParts.toList,
          sourceParts.toSet, List(), preprocessing, preprocessing.pointerNetStartState)
    } yield {
      MatchingLabel(matching.map(x => (x._1.ind, x._2.ind)).toMap)
    }
  }
  
  private def initializeRnns(computationGraph: CompGraph): Unit = {
    forwardBuilder.newGraph()
    backwardBuilder.newGraph()
    contextBuilder.newGraph()
    pointerNetInputBuilder.newGraph()
    pointerNetOutputBuilder.newGraph()
  }

  def preprocess(source: Diagram, sourceLabel: DiagramLabel, target: Diagram,
      targetPartsInTestOrder: Seq[Part], computationGraph: CompGraph): MatchingPreprocessing = {
    initializeRnns(computationGraph)
    
    val sourceFeatures = source.features.getFeatureMatrix(source.parts)
    val sourceLabelEmbeddings = sourceLabel.partLabels.map(x => getLabelEmbedding(x, computationGraph))
    val targetFeatures = target.features.getFeatureMatrix(target.parts)
    
    val sourceLstmEmbeddings = encodeFeatures(source.parts, sourceFeatures, computationGraph)
    val targetLstmEmbeddings = encodeFeatures(target.parts, targetFeatures, computationGraph)
    
    val sourceLstmEmbeddingsMatrix = concatenateCols(new ExpressionVector(sourceLstmEmbeddings.toList))

    val distanceWeights = parameter(computationGraph.getParameter(DISTANCE_WEIGHTS))
    val matchingW1 = parameter(computationGraph.getParameter(MATCHING_W1))
    val matchingB1 = parameter(computationGraph.getParameter(MATCHING_B1))
    val matchingW2 = parameter(computationGraph.getParameter(MATCHING_W2))
    val matchingL = parameter(computationGraph.getParameter(MATCHING_L))
    
    val labelW1 = parameter(computationGraph.getParameter(LABEL_W1))
    val labelB1 = parameter(computationGraph.getParameter(LABEL_B1))
    val labelW2 = parameter(computationGraph.getParameter(LABEL_W2))
    val labelB2 = parameter(computationGraph.getParameter(LABEL_B2))
    
    val contextualLstmReadInit = parameter(computationGraph.getParameter(CONTEXTUAL_LSTM_INIT))

    var contextualTargetEmbedding:Expression = null
    /*
      val targetInput = targetFeature.matching

      if (contextualLstm) {
        // TODO: dimensionality of this thing.
        var curInput = concatenate(new ExpressionVector(List(targetInput, contextualLstmReadInit)))

        val numProcessing = 2
        var h:Expression = null
        contextBuilder.start_new_sequence()
        for (i <- 0 until numProcessing) {
          h = (contextBuilder.add_input(curInput) + targetInput)

          val attentions = softmax(concatenate(new ExpressionVector(
              sourceLstmEmbeddings.map(x => dot_product(x, h)).toList)))
          val readVector = sourceLstmEmbeddingsMatrix * attentions

          curInput = concatenate(new ExpressionVector(List(targetInput, readVector)))
        }
        contextualTargetEmbedding = h
      }
     */
    
    val matchScores = for {
      sourceIndex <- (0 until sourceFeatures.length).toArray
      sourceFeature = sourceFeatures(sourceIndex)
      sourceLabelEmbedding = sourceLabelEmbeddings(sourceIndex)
      sourceLstmEmbedding = sourceLstmEmbeddings(sourceIndex)
    } yield {
      val targetSourceScores = for {
        targetIndex <- (0 until targetFeatures.length).toArray
        targetFeature = targetFeatures(targetIndex)
        targetLstmEmbedding = targetLstmEmbeddings(targetIndex)
      } yield {
        scoreSourceTargetMatch(sourceFeature, sourceLabelEmbedding, sourceLstmEmbedding,
            targetFeature, targetLstmEmbedding, contextualTargetEmbedding, distanceWeights,
            matchingW1, matchingB1, matchingW2, matchingL, labelW1, labelB1, labelW2, labelB2,
            computationGraph)
      }
      targetSourceScores
    }

    val (pointerNetEmbeddings, pointerNetState) = if (config.pointerNet) {
      encodePointerNetwork(sourceFeatures.map(x => x.matching),
          targetPartsInTestOrder.map(x => targetFeatures(x.ind).matching))
    } else {
      (null, -1)
    }

    new MatchingPreprocessing(sourceFeatures, targetFeatures, matchScores,
        pointerNetEmbeddings, pointerNetState)
  }

  private def encodeFeatures(parts: Vector[Part], features: Array[PointExpressions],
      computationGraph: CompGraph): Array[Expression] = {

    val shuffled = Random.shuffle(parts.zip(features))
    
    if (config.lstmEncode || config.contextualLstm) {
      forwardBuilder.startNewSequence()
      val forwardEmbeddings = for {
        (part, feature) <- shuffled
      } yield {
        val embedding = forwardBuilder.addInput(concatenate(new ExpressionVector(List(feature.matching, feature.xy))))
        (part, embedding)
      }
      
      backwardBuilder.startNewSequence()
      val backwardEmbeddings = for {
        (part, feature) <- shuffled.reverse
      } yield {
        val embedding = backwardBuilder.addInput(concatenate(new ExpressionVector(List(feature.matching, feature.xy))))
        (part, embedding)
      }

      val forwardSorted = forwardEmbeddings.sortBy(x => x._1.ind)
      val backwardSorted = forwardEmbeddings.sortBy(x => x._1.ind)

      val concatenated = forwardSorted.zip(backwardSorted.reverse).map(x =>
        concatenate(new ExpressionVector(List(x._1._2, x._2._2))))

      concatenated.toArray
    } else {
      features.map(x => input(0.0f))
    }
  }

  private def encodePointerNetwork(sourceInputs: Seq[Expression], targetInputsInTestOrder: Seq[Expression]): (Array[Expression], Int) = {
    // Encode target points
    pointerNetInputBuilder.startNewSequence()
    /*
    val targetEmbeddings = for {
      targetInput <- targetInputsInTestOrder
    } yield {
      pointerNetInputBuilder.addInput(targetInput)
    }
    */

    val sourceEmbeddings = for {
      sourceInput <- sourceInputs 
    } yield {
      pointerNetInputBuilder.addInput(sourceInput)
    }

    pointerNetOutputBuilder.startNewSequence(forwardBuilder.finalS())
    (sourceEmbeddings.toArray, pointerNetOutputBuilder.state())
  }
  
  private def getLabelEmbedding(partLabel: String, computationGraph: CompGraph
      ): (Expression, Expression) = {
    
    val labelLookup = computationGraph.getLookupParameter(LABEL_EMBEDDINGS)
    val labelBiasLookup = computationGraph.getLookupParameter(LABEL_BIAS_EMBEDDINGS)
    val index = if (labelDict.contains(partLabel)) {
      labelDict.getIndex(partLabel)
    } else {
      labelDict.getIndex(LABEL_UNK)
    }
    
    val labelExpression = lookup(labelLookup, index)
    val labelBias = lookup(labelBiasLookup, index)
    (labelExpression, labelBias)
  }

  private def scoreSourceTargetMatch(sourceFeature: PointExpressions,
      sourceLabelEmbedding: (Expression, Expression), sourceLstmEmbedding: Expression,
      targetFeature: PointExpressions, targetLstmEmbedding: Expression,
      contextualTargetLstmEmbedding: Expression,
      distanceWeights: Expression,  matchingW1: Expression, matchingB1: Expression,
      matchingW2: Expression, matchingL: Expression, 
      labelW1: Expression, labelB1: Expression, labelW2: Expression, labelB2: Expression,
      computationGraph: CompGraph): Expression = {
    
    var score = input(0.0f)
    // Learn a distance metric
    // val delta = sourceFeature.xy - targetFeature.xy
    // val dist = -1.0 * (transpose(delta) * distanceWeights * delta)
    // dist

    // Matching MLP
    if (config.matchingNetwork) {
      val matchingDelta = sourceFeature.matching - targetFeature.matching
      val matchingAbsDelta = rectify(matchingDelta) + rectify(-1 * matchingDelta)
      val matchingAbsScore = matchingW2 * rectify((matchingW1 * matchingAbsDelta) + matchingB1)
      score = score + matchingAbsScore
    }

    // Matching linear model
    // val matchingDelta = sourceFeature.matching - targetFeature.matching
    // val matchingAbsDelta = rectify(matchingDelta) + rectify(-1 * matchingDelta)
    // val matchingLinearScore = matchingL * matchingAbsDelta
    // matchingLinearScore

    // MLP scoring word / point match.
    if (config.partClassifier) {
      val sourceLabelClassifier = sourceLabelEmbedding._1
      val sourceLabelBias = sourceLabelEmbedding._2
      val input = concatenate(new ExpressionVector(Seq(sourceLabelClassifier,
          targetFeature.matching)))
      val labelScore = (labelW2 * rectify(labelW1 * input + labelB1)) + sourceLabelBias
      score = score + labelScore
    }
    
    if (config.lstmEncode) {
      val lstmScore = dotProduct(sourceLstmEmbedding, targetLstmEmbedding)
      score = score + lstmScore
    }

    if (config.contextualLstm) {
      // TODO
      val contextualLstmScore = dotProduct(sourceLstmEmbedding, contextualTargetLstmEmbedding)
      score = score + contextualLstmScore
    }

    if (config.nearestNeighbor) {
      val sourceNorm = sqrt(dotProduct(sourceFeature.matching, sourceFeature.matching))
      val targetNorm = sqrt(dotProduct(targetFeature.matching, targetFeature.matching))
      val sim = cdiv(dotProduct(sourceFeature.matching, targetFeature.matching), (sourceNorm * targetNorm))
      score = score + sim
    }

    score
  }

  /**
   * Gets pairs of vectors encoding the relationship between pairs of
   * parts on the source and target side of the matching. For example,
   * if the matching is [(s1 -> t1), (s2 -> t2)], this generates
   * (r(s1, s2), r(t1, t2)). 
   */
  private def getRelationVectors(matching: List[(Part, Part)],
      preprocessing: MatchingPreprocessing, f: PointExpressions => Expression): List[(Expression, Expression)] = {
    for {
      (t1, s1) <- matching
      (t2, s2) <- matching if t2 != t1
    } yield {
      val t1ToT2 = f(preprocessing.targetFeatures(t2.ind)) - f(preprocessing.targetFeatures(t1.ind))
      val s1ToS2 = f(preprocessing.sourceFeatures(s2.ind)) - f(preprocessing.sourceFeatures(s1.ind))
      (t1ToT2, s1ToS2)
    }
  }

  def getAffineGlobalScore(matching: List[(Part, Part)],
    compGraph: CompGraph, preprocessing: MatchingPreprocessing): Expression = {
    val sourceTargetExpressions = getRelationVectors(matching, preprocessing, x => x.xy)
    if (sourceTargetExpressions.size > 2) {
      // Fit a linear regression that transforms the vectors between
      // aligned source points to those of aligned target points.
      val targetMatrix = transpose(concatenateCols(new ExpressionVector(sourceTargetExpressions.map(_._1))))
      val sourceMatrix = transpose(concatenateCols(new ExpressionVector(sourceTargetExpressions.map(_._2))))
          
      val identity = input(Dim(2, 2), new FloatVector(Vector(1.0f, 0.0f, 0.0f, 1.0f)))
      val l2Regularization = 0.1f
      val regressionParams = inverse((transpose(sourceMatrix) * sourceMatrix)
        + (identity * l2Regularization)) * (transpose(sourceMatrix) * targetMatrix)

      val errors = square(targetMatrix - (sourceMatrix * regressionParams))
      sumRows(sqrt(sumRows(errors)))
    } else {
      input(0.0f)
    }
  }
  
  def getNnGlobalScore(matching: List[(Part, Part)], 
      compGraph: CompGraph, preprocessing: MatchingPreprocessing): Expression = {
    val transformW1 = parameter(compGraph.getParameter(TRANSFORM_W1))
    val deltaW1 = parameter(compGraph.getParameter(DELTA_W1))
    val deltaB1 = parameter(compGraph.getParameter(DELTA_B1))
    val deltaW2 = parameter(compGraph.getParameter(DELTA_W2))
    val deltaB2 = parameter(compGraph.getParameter(DELTA_B2))
    getNnGlobalScore(matching, compGraph, preprocessing, x => x.xy,
          transformW1, deltaW1, deltaB1, deltaW2, deltaB2)
  }

  private def getNnGlobalScore(matching: List[(Part, Part)], 
      compGraph: CompGraph, preprocessing: MatchingPreprocessing, f: PointExpressions => Expression,
      transformW1: Expression, deltaW1: Expression, deltaB1: Expression,
      deltaW2: Expression, deltaB2: Expression): Expression = {
    val sourceTargetExpressions = getRelationVectors(matching, preprocessing, f)
    val concatenatedSourceTargets = sourceTargetExpressions.map(
        x => concatenate(new ExpressionVector(List(x._1, x._2))))

    // Apply a two-layer MLP to each input
    val transformed = concatenatedSourceTargets.map(x => 
      rectify(deltaW2 * rectify((deltaW1 * x) + deltaB1) + deltaB2)
      )

    if (transformed.length > 0) {
      transformW1 * sum(new ExpressionVector(transformed))
    } else {
      input(0.0f)
    }
  }

  private def getGlobalScores(targetPart: Part, remainingArray: Array[Part],
      currentMatching: List[(Part, Part)], compGraph: CompGraph,
      preprocessing: MatchingPreprocessing): Array[Expression] = {
    
    var scoreArray = remainingArray.map(x => input(0.0f))

    if (config.affineTransformScore) {
      val currentMse = getAffineGlobalScore(currentMatching, compGraph, preprocessing)
      val affineTransformParam = parameter(compGraph.getParameter(AFFINE_TRANSFORM_PARAM))
      
      val affineScores = remainingArray.map { curSource => 
        val candidateMatching = (targetPart, curSource) :: currentMatching
        val candidateMse = getAffineGlobalScore(candidateMatching, compGraph, preprocessing)
        affineTransformParam * (candidateMse - currentMse)
      }

      scoreArray = scoreArray.zip(affineScores).map(x => x._1 + x._2)
    }
       
    if (config.structuralConsistency) {
      val transformW1 = parameter(compGraph.getParameter(TRANSFORM_W1))
      val deltaW1 = parameter(compGraph.getParameter(DELTA_W1))
      val deltaB1 = parameter(compGraph.getParameter(DELTA_B1))
      val deltaW2 = parameter(compGraph.getParameter(DELTA_W2))
      val deltaB2 = parameter(compGraph.getParameter(DELTA_B2))
      val currentNnScore = getNnGlobalScore(currentMatching, compGraph, preprocessing,
          x => x.xy, transformW1, deltaW1, deltaB1, deltaW2, deltaB2)
      
      val nnScores = remainingArray.map { curSource => 
        val candidateMatching = (targetPart, curSource) :: currentMatching
        val candidateNnScore = getNnGlobalScore(candidateMatching, compGraph, preprocessing,
            x => x.xy, transformW1, deltaW1, deltaB1, deltaW2, deltaB2)

        candidateNnScore - currentNnScore
      }
      scoreArray = scoreArray.zip(nnScores).map(x => x._1 + x._2)
    }
    
    if (config.relativeAppearance) {
      val appearanceW1 = parameter(compGraph.getParameter(APPEARANCE_W1))
      val appearanceDeltaW1 = parameter(compGraph.getParameter(APPEARANCE_DELTA_W1))
      val appearanceDeltaB1 = parameter(compGraph.getParameter(APPEARANCE_DELTA_B1))
      val appearanceDeltaW2 = parameter(compGraph.getParameter(APPEARANCE_DELTA_W2))
      val appearanceDeltaB2 = parameter(compGraph.getParameter(APPEARANCE_DELTA_B2))
      val currentNnScore = getNnGlobalScore(currentMatching, compGraph, preprocessing, x => x.matching,
          appearanceW1, appearanceDeltaW1, appearanceDeltaB1, appearanceDeltaW2, appearanceDeltaB2)
      
      val nnScores = remainingArray.map { curSource => 
        val candidateMatching = (targetPart, curSource) :: currentMatching
        val candidateNnScore = getNnGlobalScore(candidateMatching, compGraph, preprocessing,
            x => x.matching, appearanceW1, appearanceDeltaW1, appearanceDeltaB1, appearanceDeltaW2,
            appearanceDeltaB2)

        candidateNnScore - currentNnScore
      }
      scoreArray = scoreArray.zip(nnScores).map(x => x._1 + x._2)
    }

    scoreArray
  }

  /**
   * Get a score for matching targetPart against each source part
   * in remainingArray, given the currentMatching. 
   */
  private def getScores(targetPart: Part, remainingArray: Array[Part],
      currentMatching: List[(Part, Part)], compGraph: CompGraph,
      preprocessing: MatchingPreprocessing,
      pointerNetState: Int, pointerNetInput: Expression): (Expression, Int) = {
    val unaryScores = remainingArray.map(x => preprocessing.getMatchScore(x, targetPart))

    val globalScores = getGlobalScores(targetPart, remainingArray, currentMatching, compGraph,
        preprocessing)
    var scores = unaryScores.zip(globalScores).map(x => x._1 + x._2)
    
    var pointerNetNextState = -1
    if (config.pointerNet) {
      val pointerNetOutput = pointerNetOutputBuilder.addInput(pointerNetState, pointerNetInput)
      pointerNetNextState = pointerNetOutputBuilder.state()
      
      val attentionScores = remainingArray.map(x =>
        dotProduct(preprocessing.pointerNetEmbeddings(x.ind), pointerNetOutput))

      /*
      val pnSourceW = parameter(compGraph.getParameter(POINTER_NET_SOURCE_W))
      val pnTargetW = parameter(compGraph.getParameter(POINTER_NET_TARGET_W))
      val pnV = parameter(compGraph.getParameter(POINTER_NET_V))
      
      val attentionScores = remainingArray.map(x =>
        dotProduct(pnV, tanh(pnSourceW * preprocessing.pointerNetEmbeddings(x.ind))
            + (pnTargetW * pointerNetOutput)))
      */
      scores = scores.zip(attentionScores).map(x => x._1 + x._2)
    }

    (concatenate(new ExpressionVector(scores.toVector)), pointerNetNextState)
  }

  /**
   * Match each part in targetParts with a distinct part in
   * remainingSourceParts. 
   */
  private def matchRemaining(targetParts: List[Part], remainingSourceParts: Set[Part],
      previousMatching: List[(Part, Part)], preprocessing: MatchingPreprocessing,
      pointerNetState: Int): Pnp[List[(Part, Part)]] = {
    if (targetParts.length == 0) {
      Pnp.value(previousMatching)
    } else {
      val targetPart = targetParts.head
      val remainingArray = remainingSourceParts.toArray

      for {
        cg <- computationGraph()
        (scoresExpression, nextPointerNetState) = getScores(targetPart, remainingArray,
            previousMatching, cg, preprocessing, pointerNetState,
            preprocessing.targetFeatures(targetPart.ind).matching)
        chosenSourcePart <- Pnp.choose(remainingArray, scoresExpression, targetPart)
        nextSourceParts = if (config.matchIndependent) {
          remainingSourceParts
        } else {
          remainingSourceParts - chosenSourcePart
        }
        matching = (targetPart, chosenSourcePart) :: previousMatching
        
        nextPointerNetInput = if (config.pointerNet) {
          preprocessing.sourceFeatures(chosenSourcePart.ind).matching
        } else {
          null
        }

        rest <- matchRemaining(targetParts.tail, nextSourceParts, matching, preprocessing,
            nextPointerNetState)
      } yield {
        rest
      }
    }
  }

  /**
   * Gets an execution score that assigns 0 to
   * executions that are consistent with {@code label} and
   * -infinity to the rest.
   */
  def getLabelScore(label: MatchingLabel): ExecutionScore = {
    val score = MatchingExecutionScore(label)
    ((x, y, z) => if (score(x, y, z) > 0.0) { Double.NegativeInfinity } else { 0.0 })
  }

  /**
   * Gets an execution score that assigns a cost of 0 to
   * executions that are consistent with {@code label}, and
   * 1 per mistake to the rest.
   */
  def getMarginScore(label: MatchingLabel): ExecutionScore = {
    MatchingExecutionScore(label)
  }

  def save(saver: ModelSaver): Unit = {
    saver.addObject(config)
    saver.addLstmBuilder(forwardBuilder)
    saver.addLstmBuilder(backwardBuilder)
    saver.addLstmBuilder(contextBuilder)
    saver.addLstmBuilder(pointerNetInputBuilder)
    saver.addLstmBuilder(pointerNetOutputBuilder)
    saver.addObject(labelDict)
  }
}

/**
 * Stores the output of preprocessing and neural network
 * computations that can be shared across many choices.
 */
class MatchingPreprocessing(val sourceFeatures: Array[PointExpressions],
    val targetFeatures: Array[PointExpressions], val matchScores: Array[Array[Expression]],
    val pointerNetEmbeddings: Array[Expression], val pointerNetStartState: Int) {
  
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
        1.0
      }
    } else {
      0.0
    }
  }
}

case class ContextualPointExpressions(e: PointExpressions, lstm: Expression, label: Expression)

class MatchingModelConfig() extends Serializable {
  var xyFeatureDim: Int = -1
  var matchingFeatureDim: Int = -1
  var vggFeatureDim: Int = -1


  // Neural net dimensionalities.  
  var matchingHiddenDim = 512
  var transformHiddenDim1 = 32
  var transformHiddenDim2 = 32
  var deltaDim = 4
  var labelDim = 32
  var labelHidden = 32

  var lstmHiddenDim = 50
  
  var pointerNetHiddenDim = 100
  var pointerNetLstmDim = 64

  var matchIndependent: Boolean = false
  var affineTransformScore: Boolean = false
  var structuralConsistency: Boolean = false
  var matchingNetwork: Boolean = false
  var partClassifier: Boolean = false
  var relativeAppearance: Boolean = false
  var lstmEncode: Boolean = false
  var contextualLstm: Boolean = false
  var nearestNeighbor: Boolean = false
  var pointerNet: Boolean = false
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
  
  val APPEARANCE_W1 = "appearanceW1"
  val APPEARANCE_DELTA_W1 = "appearanceDeltaW1"
  val APPEARANCE_DELTA_B1 = "appearanceDeltaB1"
  val APPEARANCE_DELTA_W2 = "appearanceDeltaW2"
  val APPEARANCE_DELTA_B2 = "appearanceDeltaB2"

  val LABEL_EMBEDDINGS = "labelEmbeddings"
  val LABEL_BIAS_EMBEDDINGS = "labelBiasEmbeddings"
  val LABEL_UNK = "<unk>"
  val LABEL_W1 = "labelW1"
  val LABEL_B1 = "labelB1"
  val LABEL_W2 = "labelW2"
  val LABEL_B2 = "labelB2"
  
  val CONTEXTUAL_LSTM_INIT = "contextualLstmInit"
  
  val POINTER_NET_INIT = "pointerNetInit"
  val POINTER_NET_SOURCE_W = "pointerNetSourceW"
  val POINTER_NET_TARGET_W = "pointerNetTargetW"
  val POINTER_NET_V = "pointerNetV"
  
  /**
   * Create a MatchingModel and populate {@code model} with the
   * necessary neural network parameters. {@code partLabels} is
   * the set of labels that will have their own parameters in the
   * model.
   */
  def create(c: MatchingModelConfig, partLabels: Seq[String],
      model: PnpModel): MatchingModel = {
    model.addParameter(DISTANCE_WEIGHTS, Dim(c.xyFeatureDim, c.xyFeatureDim))

    model.addParameter(MATCHING_W1, Dim(c.matchingHiddenDim, c.matchingFeatureDim))
    model.addParameter(MATCHING_B1, Dim(c.matchingHiddenDim))
    model.addParameter(MATCHING_W2, Dim(1, c.matchingHiddenDim))
    model.addParameter(MATCHING_L, Dim(1, c.matchingFeatureDim))

    model.addParameter(AFFINE_TRANSFORM_PARAM, Dim(1))
    
    model.addParameter(TRANSFORM_W1, Dim(1, c.transformHiddenDim2))
    model.addParameter(DELTA_W1, Dim(c.transformHiddenDim1, c.deltaDim))
    model.addParameter(DELTA_B1, Dim(c.transformHiddenDim1))
    model.addParameter(DELTA_W2, Dim(c.transformHiddenDim2, c.transformHiddenDim1))
    model.addParameter(DELTA_B2, Dim(c.transformHiddenDim2))
    
    model.addParameter(APPEARANCE_W1, Dim(1, c.transformHiddenDim2))
    model.addParameter(APPEARANCE_DELTA_W1, Dim(c.transformHiddenDim1, c.matchingFeatureDim * 2))
    model.addParameter(APPEARANCE_DELTA_B1, Dim(c.transformHiddenDim1))
    model.addParameter(APPEARANCE_DELTA_W2, Dim(c.transformHiddenDim2, c.transformHiddenDim1))
    model.addParameter(APPEARANCE_DELTA_B2, Dim(c.transformHiddenDim2))

    val labelDict = IndexedList.create(partLabels.asJava)
    labelDict.add(LABEL_UNK)
    model.addLookupParameter(LABEL_EMBEDDINGS, labelDict.size, Dim(c.labelDim))
    model.addLookupParameter(LABEL_BIAS_EMBEDDINGS, labelDict.size, Dim(1))    
    model.addParameter(LABEL_W1, Dim(c.labelHidden, c.labelDim + c.matchingFeatureDim))
    model.addParameter(LABEL_B1, Dim(c.labelHidden))
    model.addParameter(LABEL_W2, Dim(1, c.labelHidden))
    // XXX: This parameter isn't used currently.
    model.addParameter(LABEL_B2, Dim(c.labelHidden))
    
    // Forward and backward RNNs for encoding the parts.
    val forwardBuilder = new LstmBuilder(1, c.matchingFeatureDim + c.xyFeatureDim, c.lstmHiddenDim, model.model)
    val backwardBuilder = new LstmBuilder(1, c.matchingFeatureDim + c.xyFeatureDim, c.lstmHiddenDim, model.model)
    
    // TODO: Set this parameter properly
    model.addParameter(CONTEXTUAL_LSTM_INIT, Dim(c.labelHidden))
    val contextualBuilder = new LstmBuilder(1, c.matchingFeatureDim, c.lstmHiddenDim, model.model)
    
    // Parameters for pointer network.
    model.addParameter(POINTER_NET_INIT, Dim(c.matchingFeatureDim))
    model.addParameter(POINTER_NET_SOURCE_W, Dim(c.pointerNetHiddenDim, c.pointerNetLstmDim))
    model.addParameter(POINTER_NET_TARGET_W, Dim(c.pointerNetHiddenDim, c.pointerNetLstmDim))
    model.addParameter(POINTER_NET_V, Dim(c.pointerNetHiddenDim))

    val pointerNetInputBuilder = new LstmBuilder(1, c.matchingFeatureDim, c.pointerNetLstmDim,
        model.model)
    val pointerNetOutputBuilder = new LstmBuilder(1, c.matchingFeatureDim, c.pointerNetLstmDim,
        model.model) 

    new MatchingModel(c, forwardBuilder, backwardBuilder, contextualBuilder, 
        pointerNetInputBuilder, pointerNetOutputBuilder,
        labelDict, model)
  }

  /**
   * Load a serialized MatchingModel.
   */
  def load(loader: ModelLoader, model: PnpModel): MatchingModel = {
    val config = loader.loadObject(classOf[MatchingModelConfig])

    val forwardLstm = loader.loadLstmBuilder()
    val backwardLstm = loader.loadLstmBuilder()
    val contextualBuilder = loader.loadLstmBuilder()
    val pointerNetInputBuilder = loader.loadLstmBuilder()
    val pointerNetOutputBuilder = loader.loadLstmBuilder()
    
    val labelDict = loader.loadObject(classOf[IndexedList[String]])

    new MatchingModel(config, forwardLstm, backwardLstm,
        contextualBuilder, pointerNetInputBuilder, pointerNetOutputBuilder,
        labelDict, model)
  }
}