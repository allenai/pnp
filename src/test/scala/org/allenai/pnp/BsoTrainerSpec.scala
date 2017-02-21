package org.allenai.pnp

import scala.collection.JavaConverters._
import org.scalatest.Matchers
import org.allenai.pnp.examples.Seq2Seq
import edu.cmu.dynet.DynetParams
import org.scalatest.FlatSpec

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._
import com.jayantkrish.jklol.util.IndexedList
import com.jayantkrish.jklol.training.NullLogFunction

class BsoTrainerSpec extends FlatSpec with Matchers {
  
  initialize(new DynetParams())
  
  val TOLERANCE = 0.01

  val rawData = Array(("hola", "hi <eos>"),
      ("como estas", "how are you <eos>"))
      
  val data = rawData.map(x => (x._1.split(" ").toList, x._2.split(" ").toList))
  
  val sourceVocab = IndexedList.create(data.flatMap(_._1).toSet.asJava)
  val targetVocab = IndexedList.create(data.flatMap(_._2).toSet.asJava)
  val endTokenIndex = targetVocab.getIndex("<eos>")
  
  val indexedData = for {
    d <- data
  } yield {
    val sourceIndexes = d._1.map(x => sourceVocab.getIndex(x)).toArray
    val targetIndexes = d._2.map(x => targetVocab.getIndex(x)).toArray
    (sourceIndexes, targetIndexes)
  }

  def getSeq2Seq(): Seq2Seq = {
    val model = PpModel.init(false)
    Seq2Seq.create(sourceVocab, targetVocab, endTokenIndex, model)
  }
  
  def runTest(seq2seq: Seq2Seq, input: String, expected: String): Unit = {    
    val inputIndexes = input.split(" ").map(x => sourceVocab.getIndex(x)).toArray
    val expectedIndexes = expected.split(" ").map(x => targetVocab.getIndex(x)).toArray
    val unconditional = seq2seq.apply(inputIndexes)
      
    val cg = ComputationGraph.getNew
    val graph = seq2seq.model.getComputationGraph(cg)

    val marginals = unconditional.beamSearch(10, 20, Env.init, null, graph, new NullLogFunction)
    
    marginals.executions.size should be > 0
    marginals.executions(0).value.toList should be (expectedIndexes.toList)
  }

  "BsoTrainerSpec" should "train seq2seq models" in {
    val seq2seq = getSeq2Seq()
    val model = seq2seq.model
    
    val examples = for {
      d <- indexedData
    } yield {
      val unconditional = seq2seq.apply(d._1)
      val oracle = seq2seq.generateExecutionOracle(d._2)
      PpExample(unconditional, unconditional, Env.init, oracle)
    }
    
    val sgd = new SimpleSGDTrainer(model.model, 0.1f, 0.01f)
    // val trainer = new LoglikelihoodTrainer(1000, 100, false, model, sgd, new NullLogFunction())
    val trainer = new BsoTrainer(1, 10, 10, model, sgd, new NullLogFunction())
    trainer.train(examples)
    
    for (d <- rawData) {
      runTest(seq2seq, d._1, d._2)
    }
  }
}

object BsoTrainerSpec {
  

  
}