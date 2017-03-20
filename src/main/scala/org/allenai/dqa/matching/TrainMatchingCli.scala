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
import joptsimple.OptionParser
import joptsimple.OptionSet
import joptsimple.OptionSpec
import org.allenai.pnp.BsoTrainer

/**
 * Command line program for training a matching model
 * from a collection of diagrams.
 */
class TrainMatchingCli extends AbstractCli {
  
  var diagramsOpt: OptionSpec[String] = null
  var diagramFeaturesOpt: OptionSpec[String] = null
  var examplesOpt: OptionSpec[String] = null
  var modelOutputOpt: OptionSpec[String] = null
  
  var matchIndependentOpt: OptionSpec[Void] = null
  var affineTransformOpt: OptionSpec[Void] = null
  var structuralFactorOpt: OptionSpec[Void] = null
  var matchingNetworkOpt: OptionSpec[Void] = null
  var partClassifierOpt: OptionSpec[Void] = null
  var relativeAppearanceOpt: OptionSpec[Void] = null
  var lstmEncodeOpt: OptionSpec[Void] = null
  var contextualLstmOpt: OptionSpec[Void] = null
  var nearestNeighborOpt: OptionSpec[Void] = null
  var pointerNetOpt: OptionSpec[Void] = null

  var loglikelihoodOpt: OptionSpec[Void] = null
  var pretrainOpt: OptionSpec[Void] = null
  var epochsOpt: OptionSpec[Integer] = null
  var beamSizeOpt: OptionSpec[Integer] = null
  
  override def initializeOptions(parser: OptionParser): Unit = {
    diagramsOpt = parser.accepts("diagrams").withRequiredArg().ofType(classOf[String]).required()
    diagramFeaturesOpt = parser.accepts("diagramFeatures").withRequiredArg().ofType(classOf[String]).required()
    examplesOpt = parser.accepts("examples").withRequiredArg().ofType(classOf[String]).required()
    modelOutputOpt = parser.accepts("modelOut").withRequiredArg().ofType(classOf[String]).required()
    
    // Flags controlling the model variant
    matchIndependentOpt = parser.accepts("matchIndependent")
    affineTransformOpt = parser.accepts("affineTransform")
    structuralFactorOpt = parser.accepts("structuralFactor")
    matchingNetworkOpt = parser.accepts("matchingNetwork")
    partClassifierOpt = parser.accepts("partClassifier")
    relativeAppearanceOpt = parser.accepts("relativeAppearance")
    lstmEncodeOpt = parser.accepts("lstmEncode")
    contextualLstmOpt = parser.accepts("contextualLstm")
    nearestNeighborOpt = parser.accepts("nearestNeighbor")
    pointerNetOpt = parser.accepts("pointerNet")

    // Flags controlling training behavior.
    loglikelihoodOpt = parser.accepts("loglikelihood")
    pretrainOpt = parser.accepts("pretrain")
    epochsOpt = parser.accepts("epochs").withRequiredArg().ofType(classOf[Integer]).defaultsTo(50)
    beamSizeOpt = parser.accepts("beamSize").withRequiredArg().ofType(classOf[Integer]).defaultsTo(5)
  }

  override def run(options: OptionSet): Unit = {
    Initialize.initialize()
    // Read and preprocess data
    val diagramFeatures = DiagramFeatures.fromJsonFile(options.valueOf(diagramFeaturesOpt)).map(
        x => (x.imageId, x)).toMap
    val diagramsAndLabels = Diagram.fromJsonFile(options.valueOf(diagramsOpt), diagramFeatures)
    val diagramsMap = diagramsAndLabels.map(x => (x._1.id, x)).toMap

    val xyFeatureDim = diagramFeatures.head._2.pointFeatures.head._2.xy.size.toInt
    val matchingFeatureDim = diagramFeatures.head._2.pointFeatures.head._2.matching.size.toInt
    val vggFeatureDim = diagramFeatures.head._2.pointFeatures.head._2.vgg0.size.toInt
    val vggAllFeatureDim = diagramFeatures.head._2.pointFeatures.head._2.vggAll.size.toInt
    
    println("xy feature dim: " + xyFeatureDim)
    println("matching feature dim: " + matchingFeatureDim)
    println("vgg feature dim: " + vggFeatureDim)
    println("vgg all feature dim: " + vggAllFeatureDim)
    
    // Read in pairs of examples for training
    val matchingExamples = MatchingExample.fromJsonFile(options.valueOf(examplesOpt), diagramsMap)
    // Sample examples for training
    // val matchingExamples = TrainMatchingCli.sampleMatchingExamples(diagramsAndLabels, 10)
    println(matchingExamples.length + " training examples.")

    val labelVocabulary = getLabelVocabulary(matchingExamples)
    
    val model = PnpModel.init(false)

    // Configure matching model.
    val config = new MatchingModelConfig()
    config.xyFeatureDim = xyFeatureDim
    config.matchingFeatureDim = matchingFeatureDim
    config.vggFeatureDim = vggFeatureDim
    config.matchIndependent = options.has(matchIndependentOpt)
    config.affineTransformScore = options.has(affineTransformOpt)
    config.structuralConsistency = options.has(structuralFactorOpt)
    config.matchingNetwork = options.has(matchingNetworkOpt)
    config.partClassifier = options.has(partClassifierOpt)
    config.relativeAppearance = options.has(relativeAppearanceOpt)
    config.lstmEncode = options.has(lstmEncodeOpt)
    config.contextualLstm = options.has(contextualLstmOpt)
    config.nearestNeighbor = options.has(nearestNeighborOpt)
    config.pointerNet = options.has(pointerNetOpt)
    
    val matchingModel = MatchingModel.create(config, labelVocabulary, model)

    if (options.has(pretrainOpt)) {
      val matchIndependent = matchingModel.config.matchIndependent
      val locallyNormalized = model.locallyNormalized
      matchingModel.config.matchIndependent = true
      model.locallyNormalized = true

      train(matchingExamples, matchingModel, options.valueOf(epochsOpt),
          options.valueOf(beamSizeOpt), true)

      matchingModel.config.matchIndependent = matchIndependent
      model.locallyNormalized = locallyNormalized
    }

    train(matchingExamples, matchingModel, options.valueOf(epochsOpt),
        options.valueOf(beamSizeOpt), options.has(loglikelihoodOpt))
    
    // Serialize model to disk.
    val saver = new ModelSaver(options.valueOf(modelOutputOpt))
    model.save(saver)
    matchingModel.save(saver)
    saver.done()
  }
  
  def getLabelVocabulary(examples: Seq[MatchingExample]): Vector[String] = {
    val partLabels = for {
      x <- examples
      p <- x.sourceLabel.partLabels
    } yield {
      (p, x.label)
    }

    // Count the number of unique diagram labels each part
    // label has occurred with.
    val partLabelCounts = partLabels.groupBy(x => x._1).map(
        p => (p._1, p._2.length))
    
    val countThreshold = 1
    partLabelCounts.filter(p => p._2 > countThreshold).map(_._1).toVector
  }

  def train(examples: Seq[MatchingExample], matchingModel: MatchingModel,
      epochs: Int, beamSize: Int, loglikelihood: Boolean): Unit = {
    val pnpExamples = for {
      x <- examples
    } yield {
      val unconditional = matchingModel.apply(x.source, x.sourceLabel, x.target)
      val oracle = if (loglikelihood) {
        matchingModel.getLabelScore(x.label)
      } else {
        matchingModel.getMarginScore(x.label)
      }
      
      PnpExample(unconditional, unconditional, Env.init, oracle) 
    }

    val learningRate = 0.01f
    val decay = 0.01f

    val model = matchingModel.model

    if (loglikelihood) {
      // Locally-normalized training
      println("Loglikelihood training...")
      model.locallyNormalized = true
      val sgd = new SimpleSGDTrainer(model.model, learningRate, decay)
      val trainer = new LoglikelihoodTrainer(epochs, 1, false, model, sgd,
          new DefaultLogFunction())
      trainer.train(pnpExamples.toList)
    } else {
      // Globally-normalized training
      println("LaSO training...")
      val sgd = new SimpleSGDTrainer(model.model, learningRate, decay)
      val trainer = new BsoTrainer(epochs, beamSize, -1, model, sgd, new DefaultLogFunction())
      trainer.train(pnpExamples.toList)
    }
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
      diagramType <- diagramTypes.toList
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
      // val second = first
        
      val (source, sourceLabel) = filteredDiagrams(first)
      val (target, targetLabel) = filteredDiagrams(second)
      MatchingExample.fromDiagrams(source, sourceLabel, target, targetLabel)
    }

    println("sampled " + examples.size + " matching examples")
    examples.toArray
  }
}
