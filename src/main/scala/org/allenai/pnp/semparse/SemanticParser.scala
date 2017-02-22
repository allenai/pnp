package org.allenai.pnp.semparse

import scala.collection.JavaConverters._
import scala.collection.mutable.HashMap
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.{ Map => MutableMap }
import scala.collection.mutable.MultiMap
import scala.collection.mutable.{ Set => MutableSet }
import scala.collection.mutable.SetBuilder

import org.allenai.pnp.CompGraph
import org.allenai.pnp.Env
import org.allenai.pnp.ExecutionScore
import org.allenai.pnp.Pnp
import org.allenai.pnp.Pnp._
import org.allenai.pnp.PnpModel

import com.google.common.base.Preconditions
import com.jayantkrish.jklol.ccg.lambda.Type
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.ccg.lambda2.StaticAnalysis
import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.dynet._
import edu.cmu.dynet.DynetScalaHelpers._
import edu.cmu.dynet.dynet_swig._

/** A parser mapping token sequences to a distribution over
  * logical forms.
  */
class SemanticParser(val actionSpace: ActionSpace, val vocab: IndexedList[String], inputDim: Int,
    hiddenDim: Int, maxVars: Int, forwardBuilder: LSTMBuilder, backwardBuilder: LSTMBuilder,
    actionBuilder: LSTMBuilder, val model: PnpModel) {

  var dropoutProb = -1.0
  
  import SemanticParser._

  private def initializeRnns(computationGraph: CompGraph): Unit = {
    val cg = computationGraph.cg
    forwardBuilder.new_graph(cg)
    backwardBuilder.new_graph(cg)
    actionBuilder.new_graph(cg)
  }

  /** Compute the input encoding of a list of tokens
    */
  def encode(tokens: Array[Int], entityLinking: EntityLinking): Pnp[InputEncoding] = {
    for {
      compGraph <- computationGraph()
      _ = initializeRnns(compGraph)
      inputEncoding = rnnEncode(compGraph, tokens)
      entityEncoding = encodeEntities(compGraph, entityLinking, tokens)
      // entityEncoding = null
    } yield {
      InputEncoding(tokens, inputEncoding._1, inputEncoding._2, inputEncoding._3,
          inputEncoding._4, entityEncoding)
    }
  }

  private def rnnEncode(computationGraph: CompGraph, tokens: Seq[Int]
    ): (ExpressionVector, Expression, Expression, Expression) = {
    val cg = computationGraph.cg
    val wordEmbeddings = computationGraph.getLookupParameter(SemanticParser.WORD_EMBEDDINGS_PARAM)
    
    val inputEmbeddings = tokens.map(x => lookup(cg, wordEmbeddings, x)).toArray

    // TODO: should we add dropout to the builders using the .set_dropout methods?
    forwardBuilder.start_new_sequence()
    val fwOutputs = ListBuffer[Expression]()
    for (inputEmbedding <- inputEmbeddings) {
      val fwOutput = forwardBuilder.add_input(inputEmbedding)
      val fwOutputDropout = if (dropoutProb > 0.0) {
        dropout(fwOutput, dropoutProb.asInstanceOf[Float])
      } else {
        fwOutput
      }
      fwOutputs += fwOutputDropout
      // fwOutputs += fwOutput
    }

    backwardBuilder.start_new_sequence()
    val bwOutputs = ListBuffer[Expression]()
    for (inputEmbedding <- inputEmbeddings.reverse) {
      val bwOutput = backwardBuilder.add_input(inputEmbedding)
      val bwOutputDropout = if (dropoutProb > 0.0) {
        dropout(bwOutput, dropoutProb.asInstanceOf[Float])
      } else {
        bwOutput
      }
      bwOutputs += bwOutputDropout
      // bwOutputs += bwOutput
    }
    
    val outputEmbeddings = fwOutputs.toArray.zip(bwOutputs.toList.reverse).map(
        x => concatenate(Array(x._1, x._2)))
 
    val sentEmbedding = concatenate(Array(fwOutputs.last, bwOutputs.last)) 
    val inputMatrix = concatenate(inputEmbeddings.map(reshape(_, Seq(1, inputDim))).toArray)
    val outputMatrix = concatenate(outputEmbeddings.map(reshape(_, Seq(1, 2 * hiddenDim))).toArray)
    
    // TODO: figure out how to initialize the decoder from both the
    // forward and backward LSTMs
    (forwardBuilder.final_s, sentEmbedding, inputMatrix, outputMatrix)
  }

  def encodeEntities(computationGraph: CompGraph,
      entityLinking: EntityLinking, tokens: Seq[Int]): MultiMap[Type, EntityEncoding] = {
    val cg = computationGraph.cg
    val wordEmbeddings = computationGraph.getLookupParameter(SemanticParser.WORD_EMBEDDINGS_PARAM)

    val output = ListBuffer[(Type, EntityEncoding)]()
    for (entityMatch <- entityLinking.bestEntityMatchesList) {
      val span = entityMatch._1
      val entity = entityMatch._2
      val name = entityMatch._3
      val score = entityMatch._4

      var lastOutput: Expression = null
      for (wordId <- name) {
        val inputEmbedding = lookup(cg, wordEmbeddings, wordId)
        if (lastOutput == null) {
          lastOutput = inputEmbedding
        } else {
          lastOutput = lastOutput + inputEmbedding
        }
      }
      Preconditions.checkState(lastOutput != null)
      
      val spanVector = new FloatVector(tokens.length)
      for (i <- 0 until tokens.length) {
        val value = if (i >= span.start && i < span.end) { 1.0 } else { 0.0 }
        spanVector.set(i, value.asInstanceOf[Float])
      }
      val spanExpression = input(cg, Seq(tokens.length), spanVector)

      output += ((entity.t, EntityEncoding(entity, lastOutput, span, spanExpression)))
    }

    SemanticParser.seqToMultimap(output)
  }
  
  def generateExpression(tokens: Array[Int], entityLinking: EntityLinking): Pnp[Expression2] = {
    for {
      state <- parse(tokens, entityLinking)
    } yield {
      state.decodeExpression
    }
  }

  /** Generate a distribution over logical forms given 
    * tokens.
    */
  def parse(tokens: Array[Int], entityLinking: EntityLinking): Pnp[SemanticParserState] = {
    for {
      // Encode input tokens using an LSTM.
      input <- encode(tokens, entityLinking)
      
      state = SemanticParserState.start
      
      // Choose the root type for the logical form given the
      // final output of the LSTM.
      rootWeights <- param(SemanticParser.ROOT_WEIGHTS_PARAM)
      rootBias <- param(SemanticParser.ROOT_BIAS_PARAM)
      rootScores = (rootWeights * input.sentEmbedding) + rootBias
      rootType <- choose(actionSpace.rootTypes, rootScores, state)
      
      // Recursively generate a logical form using an LSTM to
      // select logical form templates to expand on typed holes
      // in the partially-generated logical form.  
      cg <- computationGraph()
      expr <- parse(input, actionBuilder, cg.cg, state.addRootType(rootType))
    } yield {
      expr
    }
  }

  private def parse(input: InputEncoding, builder: RNNBuilder,
      cg: ComputationGraph, startState: SemanticParserState): Pnp[SemanticParserState] = {
    // Initialize the output LSTM before generating the logical form.
    builder.start_new_sequence(input.rnnState)
    val startRnnState = builder.state()
    
    for {
      beginActionsParam <- param(SemanticParser.BEGIN_ACTIONS + startState.unfilledHoleIds(0).t)
      e <- parse(input, builder, beginActionsParam, startRnnState, startState)
    } yield {
      e
    }
  }

  /** Recursively generates a logical form from a partial logical
    * form containing typed holes. Each application fills a single
    * hole with a template representing a constant, application or
    * lambda expression. These templates may contain additional holes
    * to be filled in the future. An LSTM is used to encode the history
    * of previously generated templates and select which template to
    * apply. 
    */
  private def parse(input: InputEncoding, builder: RNNBuilder, prevInput: Expression,
      rnnState: Int, state: SemanticParserState): Pnp[SemanticParserState] = {
    if (state.unfilledHoleIds.length == 0) {
      // If there are no holes, return the completed logical form.
      Pnp.value(state)
    } else {
      // Select the first unfilled hole and select the
      // applicable templates given the hole's type.
      val hole = state.unfilledHoleIds.head
      val actionTemplates = actionSpace.getTemplates(hole.t)
      val allVariableTemplates = hole.scope.getVariableTemplates(hole.t)
      val variableTemplates = if (allVariableTemplates.length > maxVars) {
        // The model only has parameters for MAX_VARS variables. 
        allVariableTemplates.slice(0, maxVars)
      } else {
        allVariableTemplates
      }
      val baseTemplates = actionTemplates ++ variableTemplates

      val entities = input.entityEncoding.getOrElse(hole.t, Set()).toArray
      val entityTemplates = entities.map(_.entity.template)
      val entityVectors = entities.map(_.vector)

      val allTemplates = baseTemplates ++ entityTemplates

      // Update the LSTM and use its output to score
      // the applicable templates.
      val rnnOutput = builder.add_input(rnnState, prevInput)
      val rnnOutputDropout = if (dropoutProb > 0.0) {
        dropout(rnnOutput, dropoutProb.asInstanceOf[Float])
      } else {
        rnnOutput
      }
      val nextRnnState = builder.state
      for {
        // Compute an attention vector
        attentionWeights <- param(SemanticParser.ATTENTION_WEIGHTS_PARAM)
        wordAttentions = transpose(softmax(input.encodedTokenMatrix * attentionWeights * rnnOutputDropout)) 

        // Attention vector using the input token vectors 
        // attentionVector = transpose(wordAttentions * input.tokenMatrix)
        // Attention vector using the encoded tokens 
        attentionVector = transpose(wordAttentions * input.encodedTokenMatrix)
        
        /*
        attentionActionWeights <- param(SemanticParser.ATTENTION_ACTION_WEIGHTS_PARAM + hole.t)
        attentionActionScores = attentionActionWeights * attentionVector

        actionWeights <- param(SemanticParser.ACTION_WEIGHTS_PARAM + hole.t)
        actionBias <- param(SemanticParser.ACTION_BIAS_PARAM + hole.t)
        rnnActionScores = actionWeights * rnnOutputDropout
        */

        actionHiddenWeights <- param(SemanticParser.ACTION_HIDDEN_WEIGHTS)
        actionHiddenWeights2 <- param(SemanticParser.ACTION_HIDDEN_ACTION + hole.t)
        attentionAndRnn = concatenate(Array(attentionVector, rnnOutputDropout))
        actionHidden = tanh(actionHiddenWeights * attentionAndRnn)
        actionHiddenDropout = if (dropoutProb > 0.0) {
          dropout(actionHidden, dropoutProb.asInstanceOf[Float]) 
        } else {
          actionHidden
        }
        actionHiddenScores = actionHiddenWeights2 * actionHidden
 
        // Score the templates.
        actionScores = pickrange(actionHiddenScores, 0, baseTemplates.length)

        // Score the entity templates
        entityBias <- param(SemanticParser.ENTITY_BIAS_PARAM + hole.t)
        entityWeights <- param(SemanticParser.ENTITY_WEIGHTS_PARAM + hole.t)
        allScores = if (entities.size > 0) {
          // TODO: How should we score these entities using attentions?
          /*
            entityChoiceScore = dot_product(entityWeights, rnnOutputDropout) + entityBias 
            entityScores = concatenate(entityVectors.map(v => dot_product(v, attentionVector)
                + entityChoiceScore))
           */
          
          val entityScores = concatenate(entities.map(x => wordAttentions * x.spanVector))
          concatenate(Array(actionScores, entityScores))
        } else {
          actionScores
        }

        // Nondeterministically select which template to update
        // the parser's state with. The tag for this choice is 
        // its index in the sequence of generated templates, which
        // can be used to supervise the parser.
        templateTuple <- choose(allTemplates.zipWithIndex.toArray, allScores, state)
        nextState = templateTuple._1.apply(state).addAttention(wordAttentions)

        // Get the LSTM input parameters associated with the chosen
        // template.
        cg <- computationGraph()
        actionLookup = cg.getLookupParameter(SemanticParser.ACTION_LOOKUP_PARAM + hole.t)
        entityLookup = cg.getLookupParameter(SemanticParser.ENTITY_LOOKUP_PARAM + hole.t)
        index = templateTuple._2
        actionInput = if (index < baseTemplates.length) {
          lookup(cg.cg, actionLookup, templateTuple._2)
        } else {
          lookup(cg.cg, entityLookup, 0)
        }

        // Recursively fill in any remaining holes.
        returnState <- parse(input, builder, concatenate(Array(actionInput, attentionVector)),
            nextRnnState, nextState)
      } yield {
        returnState
      }
    }
  }

  private def concatenate(exprs: Array[Expression]): Expression = {
    val expressionVector = new ExpressionVector(exprs.length)
    for (i <- 0 until exprs.length) {
      expressionVector.set(i, exprs(i))
    }
    concatenate_VE(expressionVector)
  }

  /** Generate the sequence of parser actions that produces exp.
    * This method assumes that only one such action sequence exists.  
    */
  def generateActionSequence(exp: Expression2, entityLinking: EntityLinking,
      typeDeclaration: TypeDeclaration): Option[(List[Type], List[Template])] = {
    val holeIndexMap = MutableMap[Int, Int]()
    val actionTypes = ListBuffer[Type]()
    val actions = ListBuffer[Template]()

    val typeMap = StaticAnalysis.inferTypeMap(exp, TypeDeclaration.TOP, typeDeclaration)
      .asScala.toMap

    var state = SemanticParserState.start().addRootType(typeMap(0))
    holeIndexMap(state.nextHole.get.id) = 0  
    
    while (state.nextHole.isDefined) {
      val hole = state.nextHole.get
      val expIndex = holeIndexMap(hole.id)
      val currentScope = hole.scope

      val curType = typeMap(expIndex)
      Preconditions.checkState(curType.equals(hole.t),
          "type-checked type %s does not match hole %s at index %s of %s",
          curType, hole.t, expIndex.asInstanceOf[Integer], exp)

      val templates = actionSpace.getTemplates(curType) ++
        currentScope.getVariableTemplates(curType) ++
        entityLinking.getEntitiesWithType(curType).map(_.template)
        
      val matches = templates.filter(_.matches(expIndex, exp, typeMap))
      if (matches.size != 1) {
        println("Found " + matches.size + " for expression " + exp.getSubexpression(expIndex) +
            " : "  + curType + " (expected 1)")
        return None
      }
      
      val theMatch = matches.toList(0)
      state = theMatch.apply(state)
      
      actionTypes += curType
      actions += theMatch

      var holeOffset = 0 
      for ((holeIndex, hole) <- theMatch.holeIndexes.zip(
          state.unfilledHoleIds.slice(0, theMatch.holeIndexes.length))) {
        val curIndex = expIndex + holeIndex + holeOffset
        holeIndexMap(hole.id) = curIndex
        holeOffset += (exp.getSubexpression(curIndex).size - 1)
      }
    }

    val decoded = state.decodeExpression
    Preconditions.checkState(decoded.equals(exp), "Expected %s and %s to be equal", decoded, exp)
    
    Some((actionTypes.toList, actions.toList))
  }
  
  /** Generate an execution oracle that constrains the
    * parser to generate exp. This oracle can be used 
    * to supervise the parser.
    */
  def generateExecutionOracle(exp: Expression2, entityLinking: EntityLinking,
      typeDeclaration: TypeDeclaration): Option[SemanticParserExecutionScore] = {
    for {
      (holeTypes, templates) <- generateActionSequence(exp, entityLinking, typeDeclaration)
    } yield {
      new SemanticParserExecutionScore(holeTypes.toArray, templates.toArray)
    }
  }
  
  /**
   * Serialize this parser into {@code saver}. This method
   * assumes that the {@code model} has already been 
   * serialized (using {@code model.save}).
   */
  def save(saver: ModelSaver): Unit = {
    saver.add_object(actionSpace)
    saver.add_object(vocab)
    saver.add_int(inputDim)
    saver.add_int(hiddenDim)
    saver.add_int(maxVars)
    saver.add_lstm_builder(forwardBuilder)
    saver.add_lstm_builder(backwardBuilder)
    saver.add_lstm_builder(actionBuilder)
  }
}

/** Execution score that constrains a SemanticParser
  * to generate the given rootType and sequence of
  * templates. 
  */
class SemanticParserExecutionScore(val holeTypes: Array[Type],
    val templates: Array[Template])
extends ExecutionScore {

  val rootType = holeTypes(0)
  val reversedTemplates = templates.reverse

  def apply(tag: Any, choice: Any, env: Env): Double = {
    if (tag != null && tag.isInstanceOf[SemanticParserState]) {
      val state = tag.asInstanceOf[SemanticParserState]
      if (state.numActions == 0 && state.unfilledHoleIds.length == 0) {
        // This state represents the start of parsing where we
        // choose the root type.
        Preconditions.checkArgument(choice.isInstanceOf[Type])
        if (choice.asInstanceOf[Type].equals(rootType)) {
          0.0
        } else {
          Double.NegativeInfinity
        }
      } else {
        val actionInd = state.numActions
        if (actionInd < templates.size) {
          // TODO: this test may be too inefficient.
          val chosen = choice.asInstanceOf[(Template, Int)]._1
          val myTemplates = reversedTemplates.slice(reversedTemplates.length - actionInd, reversedTemplates.length)
          if (chosen.equals(templates(actionInd)) &&
              (actionInd == 0 || state.templates.zip(myTemplates).map(x => x._1.equals(x._2)).reduce(_ && _))) { 
            0.0
          } else {
            Double.NegativeInfinity
          }
        } else {
          Double.NegativeInfinity
        }
      }
    } else {
      0.0
    }
  }
}

class MaxExecutionScore(val scores: Seq[ExecutionScore]) {
  def apply(tag: Any, choice: Any, env: Env): Double = {
    return scores.map(s => s.apply(tag, choice, env)).max
  }
}

object SemanticParser {
  
  // Parameter names used by the parser
  val WORD_EMBEDDINGS_PARAM = "wordEmbeddings"
  val ROOT_WEIGHTS_PARAM = "rootWeights"
  val ROOT_BIAS_PARAM = "rootBias"
  
  val BEGIN_ACTIONS = "beginActions:"
  val ACTION_WEIGHTS_PARAM = "actionWeights:"
  val ACTION_BIAS_PARAM = "actionBias:"
  val ACTION_LOOKUP_PARAM = "actionLookup:"
  
  val ATTENTION_WEIGHTS_PARAM = "attentionWeights:"
  val ATTENTION_ACTION_WEIGHTS_PARAM = "attentionActionWeights:"
  
  val ACTION_HIDDEN_WEIGHTS = "actionHidden"
  val ACTION_HIDDEN_ACTION = "actionHiddenOutput:"
  
  val ENTITY_BIAS_PARAM = "entityBias:"
  val ENTITY_WEIGHTS_PARAM = "entityWeights:"
  val ENTITY_LOOKUP_PARAM = "entityLookup:"
  
  def create(actionSpace: ActionSpace, vocab: IndexedList[String], model: PnpModel): SemanticParser = {
    val inputDim = 200
    val hiddenDim = 100
    val actionDim = 100
    val actionHiddenDim = 100
    val maxVars = 10

    // Initialize model
    // TODO: document these parameters.
    model.addParameter(ROOT_WEIGHTS_PARAM, Seq(actionSpace.rootTypes.length, 2 * hiddenDim))
    model.addParameter(ROOT_BIAS_PARAM, Seq(actionSpace.rootTypes.length))
    model.addParameter(ATTENTION_WEIGHTS_PARAM, Seq(2 * hiddenDim, actionDim))

    model.addParameter(ACTION_HIDDEN_WEIGHTS, Seq(actionHiddenDim, inputDim + hiddenDim))

    model.addLookupParameter(WORD_EMBEDDINGS_PARAM, vocab.size, Seq(inputDim))
    
    for (t <- actionSpace.getTypes) {
      val actions = actionSpace.getTemplates(t)
      val dim = actions.length + maxVars

      model.addParameter(BEGIN_ACTIONS + t, Seq(actionDim + inputDim))
      model.addParameter(ACTION_WEIGHTS_PARAM + t, Seq(dim, hiddenDim))
      model.addParameter(ACTION_BIAS_PARAM + t, Seq(dim))

      model.addParameter(ATTENTION_ACTION_WEIGHTS_PARAM + t, Seq(dim, inputDim))
      
      model.addParameter(ACTION_HIDDEN_ACTION + t, Seq(dim, actionHiddenDim))
      
      model.addParameter(ENTITY_BIAS_PARAM + t, Seq(1))
      model.addParameter(ENTITY_WEIGHTS_PARAM + t, Seq(hiddenDim))
      
      model.addLookupParameter(ACTION_LOOKUP_PARAM + t, dim, Seq(actionDim))
      
      model.addLookupParameter(ENTITY_LOOKUP_PARAM + t, 1, Seq(actionDim))
    }

    // Forward and backward RNNs for encoding the input token sequence
    val forwardBuilder = new LSTMBuilder(1, inputDim, hiddenDim, model.model)
    val backwardBuilder = new LSTMBuilder(1, inputDim, hiddenDim, model.model)
    // RNN for generating actions given previous actions (and the input)
    val actionBuilder = new LSTMBuilder(1, actionDim + inputDim, hiddenDim, model.model)

    new SemanticParser(actionSpace, vocab, inputDim, hiddenDim, maxVars, forwardBuilder,
        backwardBuilder, actionBuilder, model)
  }

  def load(loader: ModelLoader, model: PnpModel): SemanticParser = {
    val actionSpace = loader.load_object(classOf[ActionSpace])
    val vocab = loader.load_object(classOf[IndexedList[String]])
    val inputDim = loader.load_int()
    val hiddenDim = loader.load_int()
    val maxVars = loader.load_int()
    val forwardBuilder = loader.load_lstm_builder()
    val backwardBuilder = loader.load_lstm_builder()
    val actionBuilder = loader.load_lstm_builder()
    
    new SemanticParser(actionSpace, vocab, inputDim, hiddenDim, maxVars,
        forwardBuilder, backwardBuilder, actionBuilder, model)
  }
  
  // TODO: move this method somewhere else.
  def seqToMultimap[A, B](s: Seq[(A, B)]) = { 
    s.foldLeft(new HashMap[A, MutableSet[B]] with MultiMap[A, B]){ 
      (acc, pair) => acc.addBinding(pair._1, pair._2)
    }
  }
}

case class InputEncoding(val tokens: Array[Int], val rnnState: ExpressionVector,
    val sentEmbedding: Expression,
    val tokenMatrix: Expression, val encodedTokenMatrix: Expression,
    val entityEncoding: MultiMap[Type, EntityEncoding]) {
}

case class EntityEncoding(val entity: Entity, val vector: Expression, val span: Span,
    val spanVector: Expression) {
}

class ExpressionDecodingException extends RuntimeException {
  
}