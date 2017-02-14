package org.allenai.dqa.labeling

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

import com.jayantkrish.jklol.ccg.lambda.ExplicitTypeDeclaration
import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import com.jayantkrish.jklol.ccg.lambda2.ExpressionSimplifier
import com.jayantkrish.jklol.ccg.lambda2.SimplificationComparator
import com.jayantkrish.jklol.cli.AbstractCli
import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.dynet.dynet_swig._
import joptsimple.OptionParser
import joptsimple.OptionSet
import joptsimple.OptionSpec
import org.allenai.pnp.semparse.SemanticParser
import org.allenai.pnp.PpExample
import org.allenai.pnp.PpModel
import org.allenai.pnp.Env
import org.allenai.pnp.Pp
import com.jayantkrish.jklol.training.DefaultLogFunction
import org.allenai.pnp.LoglikelihoodTrainer
import edu.cmu.dynet.SimpleSGDTrainer
import org.allenai.pnp.semparse.ActionSpace
import com.google.common.collect.HashMultimap

class LabelingDqaCli extends AbstractCli {
  
  var diagramsOpt: OptionSpec[String] = null
  var trainingDataOpt: OptionSpec[String] = null
  
  override def initializeOptions(parser: OptionParser): Unit = {
    diagramsOpt = parser.accepts("diagrams").withRequiredArg().ofType(classOf[String]).required()
    trainingDataOpt = parser.accepts("trainingData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',').required()
  }
  
  override def run(options: OptionSet): Unit = {
    myInitialize()
  
    // Initialize expression processing for logical forms. 
    val typeDeclaration = ExplicitTypeDeclaration.getDefault
    val simplifier = ExpressionSimplifier.lambdaCalculus()
    val comparator = new SimplificationComparator(simplifier)
    
    // Read and preprocess data
    val diagramsAndLabels = Diagram.fromJsonFile(options.valueOf(diagramsOpt))
    val diagrams = diagramsAndLabels.map(_._1)
    val diagramLabels = diagramsAndLabels.map(_._2)
    val diagramMap = diagramsAndLabels.map(x => (x._1.id, x)).toMap

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
    val diagramTypes = IndexedList.create[String]
    val diagramParts = IndexedList.create[String]
    val typePartMap = HashMultimap.create[Int, Int]
    for (label <- diagramLabels) {
      val diagramTypeId = diagramTypes.add(label.diagramType)
      for (part <- label.partLabels) {
        val diagramPartId = diagramParts.add(part)
        typePartMap.put(diagramTypeId, diagramPartId)
      }
    }
    println("diagramTypes: " + diagramTypes)
    println("diagramParts: " + diagramParts)
    println("typePartMap: " + typePartMap)
    val executor = new LabelingExecutor(diagramTypes, diagramParts, typePartMap)
    
    val answerSelector = new AnswerSelector()

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
    val parser = new SemanticParser(actionSpace, vocab)

    train(trainPreprocessed, parser, executor, answerSelector)
  }

  def train(examples: Seq[PreprocessedLabelingExample], parser: SemanticParser,
      executor: LabelingExecutor, answerSelector: AnswerSelector): PpModel = {
    
    // TODO: figure out how to set this configuration in a more
    // reliable way.
    parser.dropoutProb = -1
    
    val ppExamples = for {
      ex <- examples
      denotationDist = for {
        // TODO: stage beam search?
        lf <- parser.generateExpression(ex.tokenIds, ex.entityLinking)
        denotation <- executor.execute(lf, ex.ex.diagram)
      } yield {
        denotation
      }

      unconditional = for {
        denotation <- denotationDist
        answer <- answerSelector.selectAnswer(denotation, ex.ex.answerOptions)
      } yield {
        answer
      }

      conditional = for {
        denotation <- denotationDist
        // choose the answer and condition on getting the correct answer
        // in a single search step to reduce the possibility of search errors.
        correctAnswer <- (for {
          answer <- answerSelector.selectAnswer(denotation, ex.ex.answerOptions)
          _ <- Pp.require(answer.equals(ex.ex.correctAnswer))
        } yield {
          answer
        }).inOneStep()
      } yield {
        correctAnswer
      }
    } yield {
      PpExample(unconditional, conditional, Env.init, null)
    }

    // Train model
    // TODO: need to be able to append parameters from each model.
    val model = parser.getModel

    val sgd = new SimpleSGDTrainer(model.model, 0.1f, 0.01f)
    val trainer = new LoglikelihoodTrainer(50, 100, true, model, sgd, new DefaultLogFunction())
    trainer.train(ppExamples.toList)

    model
  }
}


object LabelingDqaCli {  
  def main(args: Array[String]): Unit = {
    (new LabelingDqaCli()).run(args)
  }
}

