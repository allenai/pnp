package org.allenai.pnp.semparse

import scala.collection.JavaConverters._
import scala.collection.mutable.HashMap
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.{ Map => MutableMap }
import scala.collection.mutable.MultiMap
import scala.collection.mutable.{ Set => MutableSet }

import org.allenai.pnp.CompGraph
import org.allenai.pnp.Env
import org.allenai.pnp.ExecutionScore
import org.allenai.pnp.Pp
import org.allenai.pnp.Pp._
import org.allenai.pnp.PpModel
import org.allenai.pnp.examples.DynetScalaHelpers._

import com.google.common.base.Preconditions
import com.jayantkrish.jklol.ccg.lambda.Type
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.ccg.lambda2.StaticAnalysis
import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._

/** A parser mapping token sequences to a distribution over
  * logical forms.
  */
class SemanticParser(actionSpace: ActionSpace, vocab: IndexedList[String]) {
  
  var inputBuilder: LSTMBuilder = null
  var actionBuilder: LSTMBuilder = null

  def initializeRnns(computationGraph: CompGraph): Unit = {
    val cg = computationGraph.cg
    inputBuilder.new_graph(cg)
    actionBuilder.new_graph(cg)
  }

  /** Compute the input encoding of a list of tokens
    */
  def encode(tokens: List[Int], entityLinking: EntityLinking): Pp[InputEncoding] = {
    for {
      compGraph <- computationGraph()
      _ = initializeRnns(compGraph)
      inputEncoding = rnnEncode(inputBuilder, compGraph, tokens)
      // entityEncoding = rnnEncodeEntities(inputBuilder, compGraph, entityLinking)
      entityEncoding = null
    } yield {
      InputEncoding(inputEncoding._1, inputEncoding._2, entityEncoding)
    }
  }

  private def rnnEncode(builder: RNNBuilder, computationGraph: CompGraph, inputs: List[Int]
    ): (ExpressionVector, List[Expression]) = {
    val cg = computationGraph.cg
    val wordEmbeddings = computationGraph.getLookupParameter(SemanticParser.WORD_EMBEDDINGS_PARAM)

    builder.start_new_sequence()
    
    val outputs = ListBuffer[Expression]()
    for (input <- inputs) {
      val inputEmbedding = lookup(cg, wordEmbeddings, input)
      outputs += builder.add_input(inputEmbedding)
    }

    (builder.final_s, outputs.toList)
  }

  def rnnEncodeEntities(builder: RNNBuilder, computationGraph: CompGraph,
      entityLinking: EntityLinking): MultiMap[Type, EntityEncoding] = {
    val cg = computationGraph.cg
    val wordEmbeddings = computationGraph.getLookupParameter(SemanticParser.WORD_EMBEDDINGS_PARAM)

    val output = ListBuffer[(Type, EntityEncoding)]()
    for (entity <- entityLinking.entities) {
      for (name <- entity.names) {
        builder.start_new_sequence()
        var lastOutput: Expression = null
        for (wordId <- name) {
          val inputEmbedding = lookup(cg, wordEmbeddings, wordId)
          lastOutput = builder.add_input(inputEmbedding)
        }
        Preconditions.checkState(lastOutput != null)
        output += ((entity.t, EntityEncoding(entity, lastOutput))) 
      }
    }

    SemanticParser.seqToMultimap(output)
  }

  /** Generate a distribution over logical forms given 
    * tokens.
    */
  def generateExpression(tokens: List[Int], entityLinking: EntityLinking): Pp[Expression2] = {
    for {
      // Encode input tokens using an LSTM.
      input <- encode(tokens, entityLinking)
      
      // Choose the root type for the logical form given the
      // final output of the LSTM.
      rootWeights <- param(SemanticParser.ROOT_WEIGHTS_PARAM)
      rootBias <- param(SemanticParser.ROOT_BIAS_PARAM)
      rootScores = (rootWeights * input.tokenRnnOuts.last) + rootBias
      rootType <- choose(actionSpace.rootTypes, rootScores, -1)
      
      // Recursively generate a logical form using an LSTM to
      // select logical form templates to expand on typed holes
      // in the partially-generated logical form.  
      cg <- computationGraph()
      expr <- generateExpression(input, actionBuilder, cg.cg, rootType)
    } yield {
      expr
    }
  }

  private def generateExpression(input: InputEncoding, builder: RNNBuilder,
      cg: ComputationGraph, rootType: Type): Pp[Expression2] = {
    // Initialize the output LSTM before generating the logical form.
    builder.start_new_sequence(input.rnnState)
    val startState = builder.state()
    
    for {
      beginActionsParam <- param(SemanticParser.BEGIN_ACTIONS)
      e <- generateExpression(input, builder, beginActionsParam, startState,
          SemanticParserState.start(rootType))
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
  private def generateExpression(input: InputEncoding, builder: RNNBuilder, prevInput: Expression,
      rnnState: Int, state: SemanticParserState): Pp[Expression2] = {
    if (state.unfilledHoleIds.length == 0) {
      // If there are no holes, return the completed logical form.
      Pp.value(state.decodeExpression)
    } else {
      // Select the first unfilled hole and select the
      // applicable templates given the hole's type. 
      val (holeId, t, scope) = state.unfilledHoleIds.head
      val actionTemplates = actionSpace.getTemplates(t)
      val variableTemplates = scope.getVariableTemplates(t)

      /*
      val entities = input.entityEncoding(t)
      val entityTemplates = entities.map(_.entity.template)
      val entityVectors = entities.map(_.vector)
      */

      val allTemplates = actionSpace.getTemplates(t) ++ scope.getVariableTemplates(t)
      
      // Update the LSTM and use its output to score
      // the applicable templates.
      val rnnOutput = builder.add_input(rnnState, prevInput)
      val nextRnnState = builder.state
      for {
        // Score the templates.
        actionWeights <- param(SemanticParser.ACTION_WEIGHTS_PARAM + t)
        actionBias <- param(SemanticParser.ACTION_BIAS_PARAM + t)
        actionScores = pickrange((actionWeights * rnnOutput) + actionBias, 0, allTemplates.length)

        // Score the entity templates
        /*
        entityBias <- param(SemanticParser.ENTITY_BIAS_PARAM + t)
        entityScores = entityVectors.map(v => (v * rnnOutput) + entityBias)
        */
        
        // Nondeterministically select which template to update
        // the parser's state with. The tag for this choice is 
        // its index in the sequence of generated templates, which
        // can be used to supervise the parser.
        templateTuple <- choose(allTemplates.zipWithIndex.toArray, actionScores, state.numActions)
        nextState = templateTuple._1.apply(state)

        // Get the LSTM input parameters associated with the chosen
        // template.
        cg <- computationGraph()
        actionLookup = cg.getLookupParameter(SemanticParser.ACTION_LOOKUP_PARAM + t)
        actionInput = lookup(cg.cg, actionLookup, templateTuple._2)

        // Recursively fill in any remaining holes.
        expr <- generateExpression(input, builder, actionInput, nextRnnState, nextState)
      } yield {
        expr
      }
    }
  }

  /** Generate the sequence of parser actions that produces exp.
    * This method assumes that only one such action sequence exists.  
    */
  def generateActionSequence(exp: Expression2, typeDeclaration: TypeDeclaration): List[Template] = {
    val indexQueue = ListBuffer[(Int, Scope)]()
    indexQueue += ((0, Scope(List.empty))) 
    val actions = ListBuffer[Template]()

    val typeMap = StaticAnalysis.inferTypeMap(exp, TypeDeclaration.TOP, typeDeclaration).asScala.toMap
    var state = SemanticParserState.start(typeMap(0))
    
    while (indexQueue.size > 0) {
      val (expIndex, currentScope) = indexQueue.head
      indexQueue.remove(0)

      val curType = typeMap(expIndex)
      val templates = actionSpace.typeTemplateMap(curType) ++ currentScope.getVariableTemplates(curType)

      val matches = templates.filter(_.matches(expIndex, exp, typeMap))
      Preconditions.checkState(matches.size == 1, "Found %s matches for expression %s (expected 1)",
          matches.size.asInstanceOf[AnyRef], exp.getSubexpression(expIndex))
      val theMatch = matches.toList(0)
      state = theMatch.apply(state)
      val nextScopes = state.unfilledHoleIds.takeRight(theMatch.holeIndexes.size).map(x => x._3)
      
      actions += theMatch
      var holeOffset = 0
      for ((holeIndex, nextScope) <- theMatch.holeIndexes.zip(nextScopes)) {
        val curIndex = expIndex + holeIndex + holeOffset
        indexQueue += ((curIndex, nextScope))
        holeOffset += (exp.getSubexpression(curIndex).size - 1)
      }
    }

    val decoded = state.decodeExpression
    Preconditions.checkState(decoded.equals(exp), "Expected %s and %s to be equal", decoded, exp)
    
    actions.toList
  }
  
  /** Generate an execution oracle that constrains the
    * parser to generate exp. This oracle can be used 
    * to supervise the parser.
    */
  def generateExecutionOracle(exp: Expression2, typeDeclaration: TypeDeclaration): ExecutionScore = {
    val rootType = StaticAnalysis.inferType(exp, typeDeclaration)
    val templates = generateActionSequence(exp, typeDeclaration)
    new SemanticParserExecutionScore(rootType, templates.toArray)
  }
  
  def getModel: PpModel = {
    // TODO: I think SemanticParser should take a PpModel as 
    // a constructor argument. This implementation has a weird
    // dependence between the LSTM builders here and the
    // returned model.
    
    val inputDim = 50
    val hiddenDim = 50
    val actionDim = 50
    val maxVars = 20
    
    val names = IndexedList.create[String]
    val params = ListBuffer[Parameter]()
    val lookupNames = IndexedList.create[String]
    val lookupParams = ListBuffer[LookupParameter]()
    val model = new Model
    
    names.add(SemanticParser.ROOT_WEIGHTS_PARAM)
    params += model.add_parameters(Seq(actionSpace.rootTypes.length, hiddenDim))
    names.add(SemanticParser.ROOT_BIAS_PARAM)
    params += model.add_parameters(Seq(actionSpace.rootTypes.length))
    names.add(SemanticParser.BEGIN_ACTIONS)
    params += model.add_parameters(Seq(actionDim))
    
    lookupNames.add(SemanticParser.WORD_EMBEDDINGS_PARAM)
    lookupParams += model.add_lookup_parameters(vocab.size, Seq(inputDim))
    
    for (t <- actionSpace.getTypes) {
      val actions = actionSpace.getTemplates(t)
      val dim = actions.length + maxVars
      
      names.add(SemanticParser.ACTION_WEIGHTS_PARAM + t)
      params += model.add_parameters(Seq(dim, hiddenDim))
      names.add(SemanticParser.ACTION_BIAS_PARAM + t)
      params += model.add_parameters(Seq(dim))
      
      lookupNames.add(SemanticParser.ACTION_LOOKUP_PARAM + t)
      lookupParams += model.add_lookup_parameters(dim, Seq(actionDim))
    }
    
    inputBuilder = new LSTMBuilder(1, inputDim, hiddenDim, model)
    actionBuilder = new LSTMBuilder(1, actionDim, hiddenDim, model)

    new PpModel(names, params.toArray, lookupNames, lookupParams.toArray, model, true)
  }
}

/** A collection of templates and root types for 
  * use in a semantic parser.
  */
class ActionSpace(
    val typeTemplateMap: MultiMap[Type, Template],
    val rootTypes: Array[Type]
    ) {

  def getTemplates(t: Type): Vector[Template] = {
    typeTemplateMap.getOrElse(t, Set.empty).toVector
  }
  
  def getTypes(): Set[Type] = {
    typeTemplateMap.keySet.toSet ++ rootTypes.toSet  
  }
}

/** Execution score that constrains a SemanticParser
  * to generate the given rootType and sequence of
  * templates. 
  */
class SemanticParserExecutionScore(val rootType: Type, val templates: Array[Template])
extends ExecutionScore {
  def apply(tag: Any, choice: Any, env: Env): Double = {
    if (tag != null && tag.isInstanceOf[Int]) {
      val tagInt = tag.asInstanceOf[Int]
      if (tagInt == -1) {
        Preconditions.checkArgument(choice.isInstanceOf[Type])
        if (choice.asInstanceOf[Type].equals(rootType)) {
          0.0
        } else {
          Double.NegativeInfinity
        }
      } else if (tagInt < templates.size) {
        val chosen = choice.asInstanceOf[(Template, Int)]._1
        if (chosen.equals(templates(tagInt))) {
          0.0
        } else {
          Double.NegativeInfinity
        }
      } else {
        Double.NegativeInfinity
      }
    } else {
      0.0
    }
  }
}

object SemanticParser {
  
  // Parameter names used by the parser
  val WORD_EMBEDDINGS_PARAM = "wordEmbeddings"
  val ROOT_WEIGHTS_PARAM = "rootWeights"
  val ROOT_BIAS_PARAM = "rootBias"
  
  val BEGIN_ACTIONS = "beginActions"
  val ACTION_WEIGHTS_PARAM = "actionWeights:"
  val ACTION_BIAS_PARAM = "actionBias:"
  val ACTION_LOOKUP_PARAM = "actionLookup:"
  
  /** Create a set of templates that can generate all of
    * the logical forms in data.
    */
  def generateActionSpace(data: Seq[Expression2], typeDeclaration: TypeDeclaration): ActionSpace = {
    val applicationTemplates = for {
      x <- data
      template <- SemanticParser.generateApplicationTemplates(x, typeDeclaration) 
    } yield {
      template
    }
  
    val lambdaTemplates = for {
      x <- data
      template <- SemanticParser.generateLambdaTemplates(x, typeDeclaration) 
    } yield {
      template
    }
  
    val constantTemplates = for {
      x <- data
      typeMap = StaticAnalysis.inferTypeMap(x, TypeDeclaration.TOP, typeDeclaration).asScala
      constant <- StaticAnalysis.getFreeVariables(x).asScala
      typeInd <- StaticAnalysis.getIndexesOfFreeVariable(x, constant)
      t = typeMap(typeInd)
    } yield {
      ConstantTemplate(t, Expression2.constant(constant))
    }
    
    val rootTypes = for {
      x <- data
      typeMap = StaticAnalysis.inferTypeMap(x, TypeDeclaration.TOP, typeDeclaration).asScala
    } yield {
      typeMap(0)
    }
  
    val allTemplates = (applicationTemplates ++ lambdaTemplates ++ constantTemplates)
    val templateMap = allTemplates.map(x => (x.root, x))

    new ActionSpace(seqToMultimap(templateMap), rootTypes.toSet.toArray)
  }

  def seqToMultimap[A, B](s: Seq[(A, B)]) = { 
    s.foldLeft(new HashMap[A, MutableSet[B]] with MultiMap[A, B]){ 
      (acc, pair) => acc.addBinding(pair._1, pair._2)
    }
  }

  def generateLambdaTemplates(
      e: Expression2,
      typeDeclaration: TypeDeclaration
    ): List[LambdaTemplate] = {
    val typeMap = StaticAnalysis.inferTypeMap(e, TypeDeclaration.TOP, typeDeclaration).asScala
    val builder = ListBuffer[LambdaTemplate]()

    for (scope <- StaticAnalysis.getScopes(e).getScopes.asScala) {
      if (scope.getStart != 0) {
        val i = scope.getStart - 1

        val root = typeMap(i)
        val argTypes = StaticAnalysis.getLambdaArgumentIndexes(e, i).map(typeMap(_)).toList
        val bodyType = typeMap(StaticAnalysis.getLambdaBodyIndex(e, i))
      
        builder += LambdaTemplate(root, argTypes, bodyType)
      }
    }

    builder.toList
  }
  
  def generateApplicationTemplates(
      e: Expression2,
      typeDeclaration: TypeDeclaration
    ): List[ApplicationTemplate] = {
    val typeMap = StaticAnalysis.inferTypeMap(e, TypeDeclaration.TOP, typeDeclaration)
    val builder = ListBuffer[ApplicationTemplate]()
    generateApplicationTemplates(e, 0, typeMap.asScala, builder)
    builder.toList
  }
  
  def generateApplicationTemplates(
      e: Expression2,
      index: Int,
      typeMap: MutableMap[Integer, Type],
      builder: ListBuffer[ApplicationTemplate]
    ): Unit = {
    if (StaticAnalysis.isLambda(e, index)) {
      generateApplicationTemplates(e, StaticAnalysis.getLambdaBodyIndex(e, index),
          typeMap, builder)
    } else {
      val subexpr = e.getSubexpression(index)
      if (!subexpr.isConstant) {
        val rootType = typeMap(index)
        val subtypes = e.getChildIndexes(index).map(x => typeMap(x)).toList
        builder += ApplicationTemplate(rootType, subtypes)
      
        for (childIndex <- e.getChildIndexes(index)) {
          generateApplicationTemplates(e, childIndex, typeMap, builder)
        }
      }
    }
  }
}

case class InputEncoding(val rnnState: ExpressionVector, val tokenRnnOuts: List[Expression],
    val entityEncoding: MultiMap[Type, EntityEncoding]) {
}

case class EntityEncoding(val entity: Entity, val vector: Expression) {
}

