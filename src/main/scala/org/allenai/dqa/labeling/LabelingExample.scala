package org.allenai.dqa.labeling

import org.allenai.pnp.semparse.EntityLinking

import com.jayantkrish.jklol.util.CountAccumulator
import com.jayantkrish.jklol.util.IndexedList

import spray.json._
import spray.json.DefaultJsonProtocol._
import scala.io.Source

case class LabelingExample(val tokens: Array[String],
    val diagram: Diagram, val diagramLabel: DiagramLabel,
    val answerOptions: AnswerOptions, val correctAnswer: Int) {
  
  def preprocess(vocab: IndexedList[String]): PreprocessedLabelingExample = {
    val unkedTokens = tokens.map(
        x => if (vocab.contains(x)) { x } else { LabelingUtil.UNK })
    val tokenIds = unkedTokens.map(x => vocab.getIndex(x))

    // TODO: match ABCD labels.
    val entityLinking: EntityLinking = EntityLinking(List())

    PreprocessedLabelingExample(tokenIds, unkedTokens, entityLinking, this)
  }
}

case class PreprocessedLabelingExample(val tokenIds: Array[Int], val unkedTokens: Array[String],
    val entityLinking: EntityLinking, val ex: LabelingExample)

case class AnswerOptions(val optionTokens: Vector[Vector[String]]) {
  
  val length = optionTokens.length
  
  def matchTokens(s: String): Int = {
    // TODO: do this better.
    val indexMatches = optionTokens.zipWithIndex.map(x =>
      (x._2, x._1.filter(t => t.equals(s)).length))
    
    val best = indexMatches.maxBy(x => x._2)

    if (best._2 > 0) {
      best._1
    } else {
      -1
    }
  }
}

object LabelingExample {
  
  def fromJsonFile(filename: String, diagramMap: Map[String, (Diagram, DiagramLabel)]): Array[LabelingExample] = {
    val examples = for {
      line <- Source.fromFile(filename).getLines 
    } yield {
      fromJson(line, diagramMap)
    }

    examples.toArray
  }
  
  def fromJson(str: String, diagramMap: Map[String, (Diagram, DiagramLabel)]): LabelingExample = {
    val js = str.parseJson.asJsObject.fields
    val tokens = LabelingUtil.tokenize(js("question").asInstanceOf[JsString].value)
    val answerOptions = js("answerOptions").asInstanceOf[JsArray].elements.map(_.convertTo[String])
    val correctAnswer = js("correctAnswer").convertTo[Int]
    val diagramId = js("diagramId").convertTo[String]
    
    val answerOptionTokens = answerOptions.map(_.split(" ").toVector).toVector

    val d = diagramMap(diagramId)
    LabelingExample(tokens, d._1, d._2, AnswerOptions(answerOptionTokens),
        correctAnswer) 
  }
  
  def getWordCounts(examples: Seq[LabelingExample]): CountAccumulator[String] = {
    val acc = CountAccumulator.create[String]
    for (ex <- examples) {
      ex.tokens.map(x => acc.increment(x, 1.0)) 
    }
    acc
  }
}