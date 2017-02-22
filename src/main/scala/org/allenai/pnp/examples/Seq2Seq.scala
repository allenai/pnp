package org.allenai.pnp.examples

import com.jayantkrish.jklol.util.IndexedList
import org.allenai.pnp.PnpModel
import edu.cmu.dynet._
import edu.cmu.dynet.DynetScalaHelpers._
import edu.cmu.dynet.dynet_swig._
import scala.collection.mutable.ListBuffer
import org.allenai.pnp.CompGraph
import org.allenai.pnp.Pnp
import org.allenai.pnp.Pnp._
import org.allenai.pnp.ExecutionScore
import com.google.common.base.Preconditions
import org.allenai.pnp.Env


/**
 * Basic sequence-to-sequence model. This model encodes
 * a source token sequence with an LSTM, then generates
 * the target token sequence from an LSTM that is initialized
 * from the source LSTM. 
 */
class Seq2Seq(val sourceVocab: IndexedList[String], val targetVocab: IndexedList[String],
    val endTokenIndex: Int, forwardBuilder: LSTMBuilder, outputBuilder: LSTMBuilder,
    val model: PnpModel) {

  var dropoutProb = -1.0
  val targetVocabInds = (0 until targetVocab.size()).toArray
  
  import Seq2Seq._
  
  private def initializeRnns(computationGraph: CompGraph): Unit = {
    val cg = computationGraph.cg
    forwardBuilder.new_graph(cg)
    outputBuilder.new_graph(cg)
  }

  /**
   * Encode the source tokens with the source LSTM, returning
   * the LSTM's state.  
   */
  private def rnnEncode(computationGraph: CompGraph, tokens: Seq[Int]): ExpressionVector = {
    val cg = computationGraph.cg
    
    val wordEmbeddings = computationGraph.getLookupParameter(SOURCE_EMBEDDINGS)
    val inputEmbeddings = tokens.map(x => lookup(cg, wordEmbeddings, x)).toArray
    
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
    
    return forwardBuilder.final_s
  }
  
  /**
   * Apply this model to a sequence of source tokens to produce a
   * probabilistic neural program over target token sequences. 
   * The (distribution over) target sequences can be approximated
   * by running inference on the returned program.
   */
  def apply(sourceTokens: Seq[Int]): Pnp[List[Int]] = {
    for {
      cg <- computationGraph()
      _ = initializeRnns(cg)
      inputEncoding = rnnEncode(cg, sourceTokens)
      _ = outputBuilder.start_new_sequence(inputEncoding)
      startRnnState = outputBuilder.state()
      startInput <- param(TARGET_START_INPUT)
      output <- generateTargetTokens(0, startRnnState, startInput)
    } yield {
      output
    }
  }
  
  def generateTargetTokens(tokenNum: Int, state: Int, curInput: Expression): Pnp[List[Int]] = {
    val output = outputBuilder.add_input(state, curInput)
    val nextState = outputBuilder.state
    
    for {
      outputWeights <- param(TARGET_WEIGHTS)
      outputTokenScores = outputWeights * output
      outputTokenIndex <- choose(targetVocabInds, outputTokenScores, tokenNum)
      
      cg <- computationGraph()
      outputTokenLookups = cg.getLookupParameter(TARGET_EMBEDDINGS)
      outputTokenEmbedding = lookup(cg.cg, outputTokenLookups, outputTokenIndex)
      
      v <- if (outputTokenIndex == endTokenIndex) {
        value(List(endTokenIndex))
      } else {
        for {
          rest <- generateTargetTokens(tokenNum + 1, nextState, outputTokenEmbedding)
        } yield {
          outputTokenIndex :: rest
        }
      }
    } yield {
      v
    }
  }
  
  def generateExecutionOracle(targetTokens: Seq[Int]): Seq2SeqExecutionScore = {
    new Seq2SeqExecutionScore(targetTokens.toArray)
  }
}

class Seq2SeqExecutionScore(val targetTokens: Array[Int]) extends ExecutionScore {
  def apply(tag: Any, choice: Any, env: Env): Double = {
    if (tag != null && tag.isInstanceOf[Int]) {
      val tokenIndex = tag.asInstanceOf[Int]
      
      Preconditions.checkArgument(choice.isInstanceOf[Int])
      val chosen = choice.asInstanceOf[Int]
      if (tokenIndex < targetTokens.length && targetTokens(tokenIndex) == chosen) {
        0.0
      } else {
        Double.NegativeInfinity
      }
    } else {
      0.0
    }
  }
}

object Seq2Seq {
  val SOURCE_EMBEDDINGS = "sourceEmbeddings"
  val TARGET_EMBEDDINGS = "targetEmbeddings"
  val TARGET_START_INPUT = "targetStartInput"
  val TARGET_WEIGHTS = "targetWeights"
  
  def create(sourceVocab: IndexedList[String], targetVocab: IndexedList[String],
    endTokenIndex: Int, model: PnpModel): Seq2Seq = {

    val sourceDim = 100
    val hiddenDim = 100
    val targetDim = 100
    
    model.addLookupParameter(SOURCE_EMBEDDINGS, sourceVocab.size, Seq(sourceDim))
    model.addLookupParameter(TARGET_EMBEDDINGS, targetVocab.size, Seq(targetDim))
    
    model.addParameter(TARGET_START_INPUT, Seq(targetDim))
    model.addParameter(TARGET_WEIGHTS, Seq(targetVocab.size, hiddenDim))
    
    val sourceBuilder = new LSTMBuilder(1, sourceDim, hiddenDim, model.model)
    val targetBuilder = new LSTMBuilder(1, targetDim, hiddenDim, model.model)
    
    new Seq2Seq(sourceVocab, targetVocab, endTokenIndex, sourceBuilder, targetBuilder, model) 
  }
}