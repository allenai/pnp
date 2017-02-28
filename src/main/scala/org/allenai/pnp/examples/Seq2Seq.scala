package org.allenai.pnp.examples

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

import org.allenai.pnp.CompGraph
import org.allenai.pnp.Env
import org.allenai.pnp.ExecutionScore
import org.allenai.pnp.Pnp
import org.allenai.pnp.Pnp._
import org.allenai.pnp.PnpModel

import com.google.common.base.Preconditions
import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.dynet._
import edu.cmu.dynet.DynetScalaHelpers._
import edu.cmu.dynet.dynet_swig._
import org.allenai.pnp.PnpExample
import com.jayantkrish.jklol.training.NullLogFunction
import org.allenai.pnp.BsoTrainer

/**
 * Basic sequence-to-sequence model. This model encodes
 * a source token sequence with an LSTM, then generates
 * the target token sequence from an LSTM that is initialized
 * from the source LSTM. 
 */
class Seq2Seq[S, T](val sourceVocab: IndexedList[S], val targetVocab: IndexedList[T],
    val endTokenIndex: Int, forwardBuilder: LSTMBuilder, outputBuilder: LSTMBuilder,
    val model: PnpModel) {

  var dropoutProb = -1.0
  val targetVocabInds = (0 until targetVocab.size()).toArray
  
  import Seq2Seq._
  
  /**
   * Apply this model to a sequence of source tokens (encoded as integers)
   * to produce a probabilistic neural program over target token sequences. 
   * The (distribution over) target sequences can be approximated
   * by running inference on the returned program.
   */
  def applyEncoded(sourceTokens: Seq[Int]): Pnp[List[Int]] = {
    for {
      cg <- computationGraph()
      // Initialize the source and target LSTMs on this computation
      // graph and encode the source tokens.
      _ = initializeRnns(cg)
      inputEncoding = rnnEncode(cg, sourceTokens)
      
      // Initialize the output LSTM
      _ = outputBuilder.start_new_sequence(inputEncoding)
      startRnnState = outputBuilder.state()
      startInput <- param(TARGET_START_INPUT)

      // Generate target sequence. output represents a single
      // possible target sequence.
      output <- generateTargetTokens(0, startRnnState, startInput)
    } yield {
      output
    }
  }

  /**
   * Same as apply above, but automatically maps the source and
   * targets to their indexes. 
   */
  def apply(sourceTokens: Seq[S]): Pnp[List[T]] = {
    val sourceInts = sourceTokens.map(x => sourceVocab.getIndex(x))
    
    for {
      targetInts <- applyEncoded(sourceInts)
    } yield {
      targetInts.map(x => targetVocab.get(x))
    }
  }

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
    }
    
    return forwardBuilder.final_s
  }

  /**
   * Generate a probabilistic neural program over target tokens,
   * given the current token's index and an LSTM state and input. 
   */
  private def generateTargetTokens(tokenIndex: Int, state: Int, curInput: Expression): Pnp[List[Int]] = {
    // Run one step of the LSTM to get the next state and output. 
    val lstmOutput = outputBuilder.add_input(state, curInput)
    val nextState = outputBuilder.state
    
    for {
      // Select an action as a linear function on top of the LSTM's
      // output. outputWeights has one row per word in the target vocab.
      targetWeights <- param(TARGET_WEIGHTS)
      targetTokenScores = targetWeights * lstmOutput
      
      // Make a discrete choice of the target token. targetTokenIndex
      // represents a single possible token, but the final probabilistic
      // neural program will represent the space of all possible tokens.
      targetTokenIndex <- choose(targetVocabInds, targetTokenScores, tokenIndex)
      
      // Get the LSTM input associated with the chosen target token, which
      // is necessary to generate the next target.
      cg <- computationGraph()
      outputTokenLookups = cg.getLookupParameter(TARGET_EMBEDDINGS)
      outputTokenEmbedding = lookup(cg.cg, outputTokenLookups, targetTokenIndex)
      
      v <- if (targetTokenIndex == endTokenIndex) {
        // If we chose the end of sequence token, we're done.
        value(List(endTokenIndex))
      } else {
        // Otherwise, recursively generate the rest of the sequence,
        // add the chosen token to the front, and return that.
        for {
          rest <- generateTargetTokens(tokenIndex + 1, nextState, outputTokenEmbedding)
        } yield {
          targetTokenIndex :: rest
        }
      }
    } yield {
      v
    }
  }

  def generateLabelCost(targetTokens: Seq[T]): Seq2SeqExecutionScore = {
    new Seq2SeqExecutionScore(targetTokens.map(x => targetVocab.getIndex(x)).toArray)
  }

  def generateLabelCostEncoded(targetTokens: Seq[Int]): Seq2SeqExecutionScore = {
    new Seq2SeqExecutionScore(targetTokens.toArray)
  }
}

class Seq2SeqExecutionScore(val targetTokensLabel: Array[Int]) extends ExecutionScore {
  def apply(tag: Any, choice: Any, env: Env): Double = {
    if (tag != null && tag.isInstanceOf[Int]) {
      // The tag is the index of the choice in the target
      // sequence, and choice is the chosen token.
      // Cost is 0 if the choice agrees with the label
      // and -infinity otherwise.
      val tokenIndex = tag.asInstanceOf[Int]
      
      Preconditions.checkArgument(choice.isInstanceOf[Int])
      val chosen = choice.asInstanceOf[Int]
      if (tokenIndex < targetTokensLabel.length && targetTokensLabel(tokenIndex) == chosen) {
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
  
  /**
   * Creates a new sequence-to-sequence model given the 
   * source and target vocabularies and a model within which
   * to initialize parameters. 
   */
  def create[S,T](sourceVocab: IndexedList[S], targetVocab: IndexedList[T],
    endTokenIndex: Int, model: PnpModel): Seq2Seq[S,T] = {

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
  
  def main(args: Array[String]): Unit = {
    // Simple example showing training and testing of
    // sequence-to-sequence
    
    // Initialize dynet
    initialize(new DynetParams())
    
    // Random parameters here
    val beamSize = 5
    val maxBeamSteps = 10

    // 0. Read data and preprocess it.
    val trainingData = Array(("hola", "hi <eos>"),
        ("como estas", "how are you <eos>"))
        
    val testData = Array(("hola como estas", "hi how are you <eos>"))

    // Tokenize input
    val trainingDataTokenized = trainingData.map(x => (x._1.split(" ").toList,
        x._2.split(" ").toList))
    val testDataTokenized = testData.map(x => (x._1.split(" ").toList,
        x._2.split(" ").toList))
  
    val sourceVocab = IndexedList.create(trainingDataTokenized.flatMap(_._1).toSet.asJava)
    val targetVocab = IndexedList.create(trainingDataTokenized.flatMap(_._2).toSet.asJava)
    val endTokenIndex = targetVocab.getIndex("<eos>")
    
    // 1. Initialize neural network model. This initializes the parameters
    // of our neural network. 
    val model = PnpModel.init(false)
    val seq2seq = Seq2Seq.create(sourceVocab, targetVocab, endTokenIndex, model)

    // 2. Generate training examples.
    val trainingExamples = for {
      d <- trainingDataTokenized
    } yield {
      // Generate a probabilistic neural program over all possible target
      // sequences given the input sequence. The parameters of the neural
      // network will be trained such that the unconditionalPnp's
      // distribution is close to the label, defined below.
      val unconditionalPnp = seq2seq.apply(d._1)
      
      // Labels can be represented either as a conditional distribution
      // over correct program executions, or a cost function that assigns 
      // a cost to each program execution. In this case we're using a cost
      // function. 
      val conditionalPnp = unconditionalPnp
      val oracle = seq2seq.generateLabelCost(d._2)
      PnpExample(unconditionalPnp, conditionalPnp, Env.init, oracle)
    }
    
    // 3. Train the model. We can select both an optimization algorithm and
    // an objective function.
    val sgd = new SimpleSGDTrainer(model.model, 0.1f, 0.01f)
    
    // Train using beam search optimization, similar to LaSO.
    // This optimizes the neural network parameters such that the
    // correct target sequence stays on the beam.
    model.locallyNormalized = false
    val trainer = new BsoTrainer(50, beamSize, maxBeamSteps, model, sgd, new NullLogFunction())
    
    // Train with maximum likelihood (i.e., the usual way
    // seq2seq models are trained).
    // model.locallyNormalized = true
    // val trainer = new LoglikelihoodTrainer(50, 100, false, model, sgd, new DefaultLogFunction())
    
    trainer.train(trainingExamples)
    
    // 4. Apply the trained model to new data.
    for (d <- testDataTokenized) {
      val cg = ComputationGraph.getNew
      val graph = seq2seq.model.getComputationGraph(cg)

      // Generate the probabilistic neural program over target
      // sequences, then run inference with the trained parameters
      // to get an approximate distribution over target sequences.
      val sourcePnp = seq2seq.apply(d._1)
      val marginals = sourcePnp.beamSearch(beamSize, maxBeamSteps, Env.init, null, graph,
          new NullLogFunction)
          
      println("Source: " + d._1)
      for (ex <- marginals.executions) {
        println("  " + ex.logProb + " " + ex.value)
      }
    }
  }
}