package org.allenai.wikitables

import org.allenai.pnp.semparse.SemanticParser

import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration
import com.jayantkrish.jklol.ccg.lambda2.ExpressionComparator
import com.jayantkrish.jklol.training.DefaultLogFunction

import edu.cmu.dynet._
import org.allenai.pnp.semparse.SemanticParserLoss

class SemanticParserLogFunction(modelDir: Option[String], bestModel: Option[String],
    parser: SemanticParser, trainExamples: Seq[WikiTablesExample],
    devExamples: Seq[WikiTablesExample], devBeam: Int, firstDevEpoch: Int,
    typeDeclaration: TypeDeclaration, comparator: ExpressionComparator,
    preprocessor: LfPreprocessor) extends DefaultLogFunction {

  var currentBestDevAccuracy = 0.0

  /**
   * Save the current parser to disk.
   */
  private def saveModel(filename: String): Unit = {
    val saver = new ModelSaver(filename)
    parser.model.save(saver)
    parser.save(saver)
    saver.done()
  }
  
  /**
   * Evaluate the semantic parser's accuracy on a development set.
   */
  private def evaluateAccuracy(examples: Seq[WikiTablesExample]): SemanticParserLoss = {
    // TODO: there needs to be a better way to set the
    // train/test configuration of the parser.
    val curDropout = parser.dropoutProb
    parser.dropoutProb = -1

    val (loss, denotations) = TestWikiTablesCli.test(examples, parser, devBeam, false, false,
        typeDeclaration, comparator, preprocessor, (x:Any) => ())

    parser.dropoutProb = curDropout
    loss
  }
  
  override def notifyIterationEnd(iteration: Long) {
    if (modelDir.isDefined) {
      startTimer("save_model")
      val filename = modelDir.get + "/parser" + "_" + iteration + ".ser"
      saveModel(filename)
      stopTimer("save_model")
    }
    
    if (trainExamples.size > 0 && iteration >= firstDevEpoch) {
      startTimer("evaluate_train")
      val loss = evaluateAccuracy(trainExamples)
      logStatistic(iteration, "train accuracy", loss.accuracy)
      logStatistic(iteration, "train oracle @ " + devBeam + " accuracy", loss.oracleAccuracy)
      stopTimer("evaluate_train")
    }

    if (devExamples.size > 0 && iteration >= firstDevEpoch) {
      startTimer("evaluate_dev")
      val loss = evaluateAccuracy(devExamples)
      logStatistic(iteration, "dev accuracy", loss.accuracy)
      logStatistic(iteration, "dev oracle @ " + devBeam + " accuracy", loss.oracleAccuracy)
      stopTimer("evaluate_dev")

      if (bestModel.isDefined && loss.accuracy > currentBestDevAccuracy) {
        saveModel(bestModel.get)
      }
      currentBestDevAccuracy = Math.max(currentBestDevAccuracy, loss.accuracy)
    }

    super.notifyIterationEnd(iteration)
  }
}