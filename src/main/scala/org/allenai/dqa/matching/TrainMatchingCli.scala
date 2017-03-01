package org.allenai.dqa.matching

import scala.annotation.migration

import org.allenai.dqa.labeling.Diagram
import org.allenai.dqa.labeling.DiagramFeatures
import org.allenai.dqa.labeling.DiagramLabel
import org.allenai.pnp.Env
import org.allenai.pnp.LoglikelihoodTrainer
import org.allenai.pnp.PnpExample
import org.allenai.pnp.PnpModel

import com.jayantkrish.jklol.cli.AbstractCli
import com.jayantkrish.jklol.training.DefaultLogFunction
import com.jayantkrish.jklol.util.Pseudorandom

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._
import joptsimple.OptionParser
import joptsimple.OptionSet
import joptsimple.OptionSpec

/**
 * Command line program for training a matching model
 * from a collection of diagrams.
 */
class TrainMatchingCli extends AbstractCli {
  
  var diagramsOpt: OptionSpec[String] = null
  var diagramFeaturesOpt: OptionSpec[String] = null
  var modelOutputOpt: OptionSpec[String] = null
  
  override def initializeOptions(parser: OptionParser): Unit = {
    diagramsOpt = parser.accepts("diagrams").withRequiredArg().ofType(classOf[String]).required()
    diagramFeaturesOpt = parser.accepts("diagramFeatures").withRequiredArg().ofType(classOf[String]).required()
    modelOutputOpt = parser.accepts("modelOut").withRequiredArg().ofType(classOf[String]).required()
  }
  
  override def run(options: OptionSet): Unit = {
    initialize(new DynetParams())
    
    // Read and preprocess data
    val diagramFeatures = DiagramFeatures.fromJsonFile(options.valueOf(diagramFeaturesOpt)).map(
        x => (x.imageId, x)).toMap
    val diagramsAndLabels = Diagram.fromJsonFile(options.valueOf(diagramsOpt), diagramFeatures)
    
    // Sample diagram pairs of the same type to create
    // matching examples.
    val matchingExamples = TrainMatchingCli.sampleMatchingExamples(diagramsAndLabels, 10)
    
    val model = PnpModel.init(true)
    val matchingModel = MatchingModel.create(10, model) 

    train(matchingExamples, matchingModel)
    
    // Serialize model to disk.
    val saver = new ModelSaver(options.valueOf(modelOutputOpt))
    model.save(saver)
    matchingModel.save(saver)
    saver.done()
  }

  def train(examples: Seq[MatchingExample], matchingModel: MatchingModel): Unit = {
    val pnpExamples = for {
      x <- examples
    } yield {
      val unconditional = matchingModel.apply(x.source, x.target)
      val oracle = matchingModel.getLabelOracle(x.label)
      PnpExample(unconditional, unconditional, Env.init, oracle) 
    }

    val model = matchingModel.model
    val sgd = new SimpleSGDTrainer(model.model, 0.1f, 0.01f)
    val trainer = new LoglikelihoodTrainer(50, 100, false, model, sgd, new DefaultLogFunction())
    trainer.train(pnpExamples.toList)
  }
}

object TrainMatchingCli {  
  def main(args: Array[String]): Unit = {
    (new TrainMatchingCli()).run(args)
  }
  
  /**
   * Samples matching examples -- pairs of diagrams of the same type -- given
   * a collection of labeled diagrams.
   */
  def sampleMatchingExamples(labeledDiagrams: Array[(Diagram, DiagramLabel)],
      examplesPerType: Int): Array[MatchingExample] = {
    val diagramTypes = labeledDiagrams.map(x => x._2.diagramType).toSet
    println(diagramTypes.size + " diagram types")
    
    val random = Pseudorandom.get()
    
    val examples = for {
      diagramType <- diagramTypes
      diagrams = labeledDiagrams.filter(_._2.diagramType.equals(diagramType))

      // throw away diagrams that don't have the full label set.
      partLabels = diagrams.flatMap(x => x._2.partLabels).toSet
      filteredDiagrams = diagrams.filter(_._2.partLabels.length == partLabels.size)

      _ = if (diagrams.length != filteredDiagrams.length) {
        println("WARNING invalid diagrams of type: " + diagramType + " orig: " + diagrams.length + " filtered: " +  filteredDiagrams.length)
      } else {
        ()
      }
      
      i <- 0 until examplesPerType
    } yield {
      val first = random.nextInt(filteredDiagrams.size)
      val second = {
        val draw = random.nextInt(filteredDiagrams.size - 1)
        if (draw >= first) {
          draw + 1 
        } else {
          draw
        }
      }
        
      val (source, sourceLabel) = filteredDiagrams(first)
      val (target, targetLabel) = filteredDiagrams(second)
      MatchingExample.fromDiagrams(source, sourceLabel, target, targetLabel)
    }

    println("sampled " + examples.size + " matching examples")
    examples.toArray
  }
}
