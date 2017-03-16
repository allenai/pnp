package org.allenai.dqa.labeling

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

import com.jayantkrish.jklol.ccg.lambda.ExplicitTypeDeclaration
import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import com.jayantkrish.jklol.ccg.lambda2.ExpressionSimplifier
import com.jayantkrish.jklol.ccg.lambda2.SimplificationComparator
import com.jayantkrish.jklol.cli.AbstractCli
import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.dynet._
import joptsimple.OptionParser
import joptsimple.OptionSet
import joptsimple.OptionSpec
import org.allenai.pnp.semparse.SemanticParser
import org.allenai.pnp.PnpExample
import org.allenai.pnp.PnpModel
import org.allenai.pnp.Env
import org.allenai.pnp.Pnp
import com.jayantkrish.jklol.training.DefaultLogFunction
import org.allenai.pnp.LoglikelihoodTrainer
import org.allenai.pnp.semparse.ActionSpace
import com.google.common.collect.HashMultimap

class LabelingDqaCli extends AbstractCli {
  
  var diagramsOpt: OptionSpec[String] = null
  var diagramFeaturesOpt: OptionSpec[String] = null
  var trainingDataOpt: OptionSpec[String] = null
  
  override def initializeOptions(parser: OptionParser): Unit = {
    diagramsOpt = parser.accepts("diagrams").withRequiredArg().ofType(classOf[String]).required()
    diagramFeaturesOpt = parser.accepts("diagramFeatures").withRequiredArg().ofType(classOf[String]).required()
    trainingDataOpt = parser.accepts("trainingData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',').required()
  }
  
  override def run(options: OptionSet): Unit = {
    Initialize.initialize()
  
    // Initialize expression processing for logical forms. 
    val typeDeclaration = ExplicitTypeDeclaration.getDefault
    val simplifier = ExpressionSimplifier.lambdaCalculus()
    val comparator = new SimplificationComparator(simplifier)
    
    // Read and preprocess data
    val diagramFeatures = DiagramFeatures.fromJsonFile(options.valueOf(diagramFeaturesOpt)).map(
        x => (x.imageId, x)).toMap
    val diagramsAndLabels = Diagram.fromJsonFile(options.valueOf(diagramsOpt), diagramFeatures)
    val diagrams = diagramsAndLabels.map(_._1)
    val diagramLabels = diagramsAndLabels.map(_._2)
    val diagramMap = diagramsAndLabels.map(x => (x._1.id, x)).toMap
    val partFeatureDim = diagramFeatures.head._2.pointFeatures.head._2.size.toInt

    val trainingData = ListBuffer[LabelingExample]()
    for (filename <- options.valuesOf(trainingDataOpt).asScala) {
      trainingData ++= LabelingExample.fromJsonFile(filename, diagramMap)
    }
    
    println(trainingData.size + " training examples")
    val wordCounts = LabelingExample.getWordCounts(trainingData)
    
    // Vocab consists of all words that appear more than once in
    // the training data.
    val vocab = IndexedList.create(wordCounts.getKeysAboveCountThreshold(1.9))
    vocab.add(LabelingUtil.UNK)
    
    val trainPreprocessed = trainingData.map(_.preprocess(vocab))

    // Configure executor for the labeling question domain theory

    /*
    println("diagramTypes: " + diagramTypes)
    println("diagramParts: " + diagramParts)
    println("typePartMap: " + typePartMap)
    */
    val model = PnpModel.init(true)
    val executor = LabelingExecutor.fromLabels(diagramLabels, partFeatureDim, model)

    // Configure semantic parser
    val actionSpace: ActionSpace = ActionSpace.fromLfConstants(executor.bindings.keySet,
        typeDeclaration)
    println("parser root types: " + actionSpace.rootTypes)
    println("parser actions: ")
    for (t <- actionSpace.typeTemplateMap.keys) {
      println(t + " ->")
      for (template <- actionSpace.typeTemplateMap.get(t)) {
        println("  " + template)
      }
    }

    val parser = SemanticParser.create(actionSpace, vocab, model)
    val answerSelector = new AnswerSelector()
    val p3 = new LabelingP3Model(parser, executor, answerSelector)

    validateParser(trainPreprocessed, parser)
    train(trainPreprocessed, p3)
    test(trainPreprocessed, p3, model)
  }
  
  def validateParser(examples: Seq[PreprocessedLabelingExample], parser: SemanticParser): Unit = {
    for (ex <- examples) {
      ComputationGraph.renew()
      val lfDist = parser.generateExpression(ex.tokenIds, ex.entityLinking)
      val dist = lfDist.beamSearch(100, 100, Env.init, parser.model.getComputationGraph(), null)
      println(ex.ex.tokens.mkString(" "))
      for (x <- dist.executions) {
        println("  "  + x)
      }
    }
  }

  def train(examples: Seq[PreprocessedLabelingExample], p3: LabelingP3Model): PnpModel = {

    // TODO: figure out how to set this configuration in a more
    // reliable way.
    p3.parser.dropoutProb = -1

    val pnpExamples = examples.map(p3.exampleToPnpExample(_))

    // Train model
    val model = p3.getModel

    val sgd = new SimpleSGDTrainer(model.model, 0.1f, 0.01f)
    val trainer = new LoglikelihoodTrainer(50, 100, true, model, sgd, new DefaultLogFunction())
    trainer.train(pnpExamples.toList)

    model
  }
  
  def test(examples: Seq[PreprocessedLabelingExample], p3: LabelingP3Model,
      model: PnpModel): Unit = {
    var numCorrect = 0 
    for (ex <- examples) {
      ComputationGraph.renew()
      val pp = p3.exampleToPnpExample(ex).unconditional
      val dist = pp.beamSearch(100, 100, Env.init, model.getComputationGraph(), null)

      println(ex.ex.tokens.mkString(" "))
      println(ex.ex.answerOptions)
      val marginals = dist.marginals
      for (x <- marginals.getSortedKeys.asScala) {
        println("  "  + x + " " + marginals.getProbability(x))
      }

      if (marginals.getSortedKeys.size > 0) {
        val bestAnswer = marginals.getSortedKeys.get(0)
        if (bestAnswer == ex.ex.correctAnswer) {
          numCorrect += 1
        }
      }
    }
    
    val accuracy = numCorrect.asInstanceOf[Double] / examples.length
    println("Accuracy: " + accuracy + " (" + numCorrect + " / " + examples.length + ")")
  }
}


object LabelingDqaCli {  
  def main(args: Array[String]): Unit = {
    (new LabelingDqaCli()).run(args)
  }
}

