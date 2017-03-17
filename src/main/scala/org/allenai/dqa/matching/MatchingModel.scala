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
import edu.cmu.dynet.DyNetScalaHelpers.RichExpression
import edu.cmu.dynet.DyNetScalaHelpers.RichNumeric
import edu.cmu.dynet.DyNetScalaHelpers.convertExpressionsToExpressions
import edu.cmu.dynet.DyNetScalaHelpers.convertFloatsToFloats
import edu.cmu.dynet.DyNetScalaHelpers.seqToDim
import edu.cmu.dynet.dynet_swig._
import scala.util.Random
import org.allenai.dqa.labeling.DiagramFeatures

/**
 * Structured prediction model for diagram part matching.
 * Given two diagrams with parts, this model predicts a
 * distribution over matchings between the two sets of
 * parts. 
 */
class MatchingModel(var matchIndependent: Boolean,
    val globalNn: Boolean, val matchingNetwork: Boolean, val partClassifier: Boolean,
    val relativeAppearance: Boolean,
    val lstmEncode: Boolean, val contextualLstm: Boolean, val nearestNeighbor: Boolean,
    val forwardBuilder: LSTMBuilder, val backwardBuilder: LSTMBuilder,
    val contextBuilder: LSTMBuilder, val labelDict: IndexedList[String], val model: PnpModel) {

  import MatchingModel._
  
  def apply(source: Diagram, sourceLabel: DiagramLabel, target: Diagram): Pnp[MatchingLabel] = {
    val sourceParts = source.parts

    for {
      cg <- computationGraph()
      preprocessing = preprocess(source, sourceLabel, target, cg)
      matching <- matchRemaining(Random.shuffle(target.parts).toList,
          sourceParts.toSet, List(), preprocessing)
    } yield {
      MatchingLabel(matching.map(x => (x._1.ind, x._2.ind)).toMap)
    }
  }
  
  private def initializeRnns(computationGraph: CompGraph): Unit = {
    val cg = computationGraph.cg
    forwardBuilder.new_graph(cg)
    backwardBuilder.new_graph(cg)
  }

  def preprocess(source: Diagram, sourceLabel: DiagramLabel, target: Diagram,
      computationGraph: CompGraph): MatchingPreprocessing = {
    initializeRnns(computationGraph)
    
    val cg = computationGraph.cg
    val sourceFeatures = source.features.getFeatureMatrix(source.parts, cg)
    val sourceLabelEmbeddings = sourceLabel.partLabels.map(x => getLabelEmbedding(x, computationGraph))
    val targetFeatures = target.features.getFeatureMatrix(target.parts, cg)
    
    val sourceLstmEmbeddings = encodeFeatures(source.parts, sourceFeatures, computationGraph)
    val targetLstmEmbeddings = encodeFeatures(target.parts, targetFeatures, computationGraph)
    
    val sourceLstmEmbeddingsMatrix = concatenate_cols(new ExpressionVector(sourceLstmEmbeddings.toList))
    
    val distanceWeights = parameter(cg, computationGraph.getParameter(DISTANCE_WEIGHTS))
    val matchingW1 = parameter(cg, computationGraph.getParameter(MATCHING_W1))
    val matchingB1 = parameter(cg, computationGraph.getParameter(MATCHING_B1))
    val matchingW2 = parameter(cg, computationGraph.getParameter(MATCHING_W2))
    val matchingL = parameter(cg, computationGraph.getParameter(MATCHING_L))
    
    val labelW1 = parameter(cg, computationGraph.getParameter(LABEL_W1))
    val labelB1 = parameter(cg, computationGraph.getParameter(LABEL_B1))
    val labelW2 = parameter(cg, computationGraph.getParameter(LABEL_W2))
    val labelB2 = parameter(cg, computationGraph.getParameter(LABEL_B2))
    
    val contextualLstmReadInit = parameter(cg, computationGraph.getParameter(CONTEXTUAL_LSTM_INIT))
    
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

    new MatchingPreprocessing(sourceFeatures, targetFeatures, matchScores)
  }

  private def encodeFeatures(parts: Vector[Part], features: Array[PointExpressions],
      computationGraph: CompGraph): Array[Expression] = {
    
    val shuffled = Random.shuffle(parts.zip(features))
    
    if (lstmEncode || contextualLstm) {
      forwardBuilder.start_new_sequence()
      val forwardEmbeddings = for {
        (part, feature) <- shuffled
      } yield {
        val embedding = forwardBuilder.add_input(feature.matching)
        (part, embedding)
      }
      
      backwardBuilder.start_new_sequence()
      val backwardEmbeddings = for {
        (part, feature) <- shuffled.reverse
      } yield {
        val embedding = backwardBuilder.add_input(feature.matching)
        (part, embedding)
      }

      val forwardSorted = forwardEmbeddings.sortBy(x => x._1.ind)
      val backwardSorted = forwardEmbeddings.sortBy(x => x._1.ind)

      val concatenated = forwardSorted.zip(backwardSorted.reverse).map(x =>
        concatenate(new ExpressionVector(List(x._1._2, x._2._2))))

      concatenated.toArray
    } else {
      features.map(x => input(computationGraph.cg, 0.0f))
    }
  }

  
  private def getLabelEmbedding(partLabel: String, computationGraph: CompGraph
      ): (Expression, Expression) = {
    val cg = computationGraph.cg
    val labelLookup = computationGraph.getLookupParameter(LABEL_EMBEDDINGS)
    val labelBiasLookup = computationGraph.getLookupParameter(LABEL_BIAS_EMBEDDINGS)
    val index = if (labelDict.contains(partLabel)) {
      labelDict.getIndex(partLabel)
    } else {
      labelDict.getIndex(LABEL_UNK)
    }
    
    val labelExpression = lookup(cg, labelLookup, index)
    val labelBias = lookup(cg, labelBiasLookup, index)
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
    
    var score = input(computationGraph.cg, 0.0f)
    // Learn a distance metric
    // val delta = sourceFeature.xy - targetFeature.xy
    // val dist = -1.0 * (transpose(delta) * distanceWeights * delta)
    // dist

    // Matching MLP
    if (matchingNetwork) {
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
    if (partClassifier) {
      val sourceLabelClassifier = sourceLabelEmbedding._1
      val sourceLabelBias = sourceLabelEmbedding._2
      val input = concatenate(new ExpressionVector(Seq(sourceLabelClassifier,
          targetFeature.matching)))
      val labelScore = (labelW2 * rectify(labelW1 * input + labelB1)) + sourceLabelBias
      score = score + labelScore
    }
    
    if (lstmEncode) {
      val lstmScore = dot_product(sourceLstmEmbedding, targetLstmEmbedding)
      score = score + lstmScore
    }

    if (contextualLstm) {
      // TODO
      val contextualLstmScore = dot_product(sourceLstmEmbedding, contextualTargetLstmEmbedding)
      score = score + contextualLstmScore
    }

    if (nearestNeighbor) {
      val sourceNorm = sqrt(dot_product(sourceFeature.matching, sourceFeature.matching))
      val targetNorm = sqrt(dot_product(targetFeature.matching, targetFeature.matching))
      val sim = cdiv(dot_product(sourceFeature.matching, targetFeature.matching), (sourceNorm * targetNorm))
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
  
  def getNnGlobalScore(matching: List[(Part, Part)], 
      compGraph: CompGraph, preprocessing: MatchingPreprocessing): Expression = {
    val transformW1 = parameter(compGraph.cg, compGraph.getParameter(TRANSFORM_W1))
    val deltaW1 = parameter(compGraph.cg, compGraph.getParameter(DELTA_W1))
    val deltaB1 = parameter(compGraph.cg, compGraph.getParameter(DELTA_B1))
    val deltaW2 = parameter(compGraph.cg, compGraph.getParameter(DELTA_W2))
    val deltaB2 = parameter(compGraph.cg, compGraph.getParameter(DELTA_B2))
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
      input(compGraph.cg, 0.0f)
    }
  }

  private def getGlobalScores(targetPart: Part, remainingArray: Array[Part],
      currentMatching: List[(Part, Part)], compGraph: CompGraph,
      preprocessing: MatchingPreprocessing): Array[Expression] = {
    
    var scoreArray = remainingArray.map(x => input(compGraph.cg, 0.0f))

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
    
    if (globalNn) {
      val transformW1 = parameter(compGraph.cg, compGraph.getParameter(TRANSFORM_W1))
      val deltaW1 = parameter(compGraph.cg, compGraph.getParameter(DELTA_W1))
      val deltaB1 = parameter(compGraph.cg, compGraph.getParameter(DELTA_B1))
      val deltaW2 = parameter(compGraph.cg, compGraph.getParameter(DELTA_W2))
      val deltaB2 = parameter(compGraph.cg, compGraph.getParameter(DELTA_B2))
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
    
    if (relativeAppearance) {
      val appearanceW1 = parameter(compGraph.cg, compGraph.getParameter(APPEARANCE_W1))
      val appearanceDeltaW1 = parameter(compGraph.cg, compGraph.getParameter(APPEARANCE_DELTA_W1))
      val appearanceDeltaB1 = parameter(compGraph.cg, compGraph.getParameter(APPEARANCE_DELTA_B1))
      val appearanceDeltaW2 = parameter(compGraph.cg, compGraph.getParameter(APPEARANCE_DELTA_W2))
      val appearanceDeltaB2 = parameter(compGraph.cg, compGraph.getParameter(APPEARANCE_DELTA_B2))
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
      preprocessing: MatchingPreprocessing): Expression = {
    val unaryScores = remainingArray.map(x => preprocessing.getMatchScore(x, targetPart))

    val globalScores = getGlobalScores(targetPart, remainingArray, currentMatching, compGraph,
        preprocessing)
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
    saver.add_boolean(matchIndependent)
    saver.add_boolean(globalNn)
    saver.add_boolean(matchingNetwork)
    saver.add_boolean(partClassifier)
    saver.add_boolean(relativeAppearance)
    saver.add_boolean(lstmEncode)
    saver.add_boolean(contextualLstm)
    saver.add_boolean(nearestNeighbor)
    saver.add_lstm_builder(forwardBuilder)
    saver.add_lstm_builder(backwardBuilder)
    saver.add_lstm_builder(contextBuilder)
    saver.add_object(labelDict)
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
        1.0
      }
    } else {
      0.0
    }
  }
}

case class ContextualPointExpressions(e: PointExpressions, lstm: Expression, label: Expression)

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

  // Neural net dimensionalities.
  // TODO: these should be configurable.
  val matchingHiddenDim = 512
  val transformHiddenDim1 = 32
  val transformHiddenDim2 = 32
  val deltaDim = 4
  val labelDim = 32
  val labelHidden = 32

  val lstmHiddenDim = 50
  
  /**
   * Create a MatchingModel and populate {@code model} with the
   * necessary neural network parameters. {@code partLabels} is
   * the set of labels that will have their own parameters in the
   * model.
   */
  def create(xyFeatureDim: Int, matchingFeatureDim: Int,
      vggFeatureDim: Int, matchIndependent: Boolean,
      globalNn: Boolean, matchingNetwork: Boolean, partClassifier: Boolean,
      relativeAppearance: Boolean, lstmEncode: Boolean, contextualLstm: Boolean,
      nearestNeighbors: Boolean, partLabels: Seq[String], model: PnpModel): MatchingModel = {
    model.addParameter(DISTANCE_WEIGHTS, Seq(xyFeatureDim, xyFeatureDim))

    model.addParameter(MATCHING_W1, Seq(matchingHiddenDim, matchingFeatureDim))
    model.addParameter(MATCHING_B1, Seq(matchingHiddenDim))
    model.addParameter(MATCHING_W2, Seq(1, matchingHiddenDim))
    model.addParameter(MATCHING_L, Seq(1, matchingFeatureDim))

    model.addParameter(AFFINE_TRANSFORM_PARAM, Seq(1))
    
    model.addParameter(TRANSFORM_W1, Seq(1, transformHiddenDim2))
    model.addParameter(DELTA_W1, Seq(transformHiddenDim1, deltaDim))
    model.addParameter(DELTA_B1, Seq(transformHiddenDim1))
    model.addParameter(DELTA_W2, Seq(transformHiddenDim2, transformHiddenDim1))
    model.addParameter(DELTA_B2, Seq(transformHiddenDim2))
    
    model.addParameter(APPEARANCE_W1, Seq(1, transformHiddenDim2))
    model.addParameter(APPEARANCE_DELTA_W1, Seq(transformHiddenDim1, matchingFeatureDim * 2))
    model.addParameter(APPEARANCE_DELTA_B1, Seq(transformHiddenDim1))
    model.addParameter(APPEARANCE_DELTA_W2, Seq(transformHiddenDim2, transformHiddenDim1))
    model.addParameter(APPEARANCE_DELTA_B2, Seq(transformHiddenDim2))

    val labelDict = IndexedList.create(partLabels.asJava)
    labelDict.add(LABEL_UNK)
    model.addLookupParameter(LABEL_EMBEDDINGS, labelDict.size, Seq(labelDim))
    model.addLookupParameter(LABEL_BIAS_EMBEDDINGS, labelDict.size, Seq(1))    
    model.addParameter(LABEL_W1, Seq(labelHidden, labelDim + matchingFeatureDim))
    model.addParameter(LABEL_B1, Seq(labelHidden))
    model.addParameter(LABEL_W2, Seq(1, labelHidden))
    // XXX: This parameter isn't used currently.
    model.addParameter(LABEL_B2, Seq(labelHidden))
    
    // Forward and backward RNNs for encoding the parts.
    val forwardBuilder = new LSTMBuilder(1, matchingFeatureDim, lstmHiddenDim, model.model)
    val backwardBuilder = new LSTMBuilder(1, matchingFeatureDim, lstmHiddenDim, model.model)
    
    // TODO: Set this parameter properly
    model.addParameter(CONTEXTUAL_LSTM_INIT, Seq(labelHidden))
    val contextualBuilder = new LSTMBuilder(1, matchingFeatureDim, lstmHiddenDim, model.model)

    new MatchingModel(matchIndependent, globalNn, matchingNetwork, partClassifier, relativeAppearance,
        lstmEncode, contextualLstm, nearestNeighbors,
        forwardBuilder, backwardBuilder, contextualBuilder, labelDict, model)
  }

  /**
   * Load a serialized MatchingModel.
   */
  def load(loader: ModelLoader, model: PnpModel): MatchingModel = {
    val matchIndependent = loader.load_boolean()
    val globalNn = loader.load_boolean()
    val matchingNetwork = loader.load_boolean()
    val partClassifier = loader.load_boolean()
    val relativeAppearance = loader.load_boolean()
    val lstmEncode = loader.load_boolean()
    val contextualLstm = loader.load_boolean()
    val nearestNeighbor = loader.load_boolean()
    val forwardLstm = loader.load_lstm_builder()
    val backwardLstm = loader.load_lstm_builder()
    val contextualBuilder = loader.load_lstm_builder()
    val labelDict = loader.load_object(classOf[IndexedList[String]])

    new MatchingModel(matchIndependent, globalNn, matchingNetwork, partClassifier,
        relativeAppearance, lstmEncode, contextualLstm, nearestNeighbor, forwardLstm, backwardLstm,
        contextualBuilder, labelDict, model)
  }
}