package org.allenai.pnp.semparse

import scala.collection.mutable.MultiMap

import com.jayantkrish.jklol.ccg.lambda.Type
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import scala.collection.mutable.ListBuffer

case class EntityLinking(matches: List[(Option[Span], Entity, List[Int], Double)]) {
  val entities: List[Entity] = matches.map(_._2).toSet.toList
  /*
  val linkedMatches: List[(Span, Entity, List[Int], Double)] =
    matches.filter(x => x._1 != None).map(x => (x._1.get, x._2, x._3, x._4))
  val unlinkedMatches: List[(Entity, List[Int], Double)] =
    matches.filter(x => x._1 == None).map(x => (x._2, x._3, x._4))
   */
  val entityMatches = SemanticParser.seqToMultimap(
      matches.map(x => (x._2, (x._1, x._3, x._4))))
  // Find matches with max score.
  val bestEntityMatches = entityMatches.map(x => (x._1, x._2.maxBy(_._3)))
  val bestEntityMatchesList = bestEntityMatches.map(x => (x._2._1, x._1, x._2._2, x._2._3)).toList
  
  val entityTypes = entities.map(_.t).toSet
  val entitiesWithType = entityTypes.map(t => (t, entities.filter(_.t.equals(t)).toArray)).toMap
  
  def getEntitiesWithType(t: Type): Array[Entity] = {
    entitiesWithType.getOrElse(t, Array())
  }
  
  def getBestEntitySpan(e: Entity): Option[Span] = {
    for {
      m <- bestEntityMatches.get(e)
      s <- m._1
    } yield {
      s
    }
  }
}

case class Entity(val expr: Expression2, val t: Type,
    val template: Template, val names: List[List[Int]]) {

  val nameTokens = names.flatten.toArray
  val nameTokensSet = names.flatten.toSet
}

class EntityDict(val map: MultiMap[List[Int], Entity]) {
  
  def lookup(tokenIds: List[Int]): Set[(Entity, List[Int], Double)] = {
    if (map.contains(tokenIds)) {
      map(tokenIds).map(x => (x, tokenIds, tokenIds.length.asInstanceOf[Double])).toSet
    } else {
      Set()
    }
  }

  def link(tokenIds: List[Int]): EntityLinking = {
    // This is a naive way to match entity names against the 
    // question text, but it's probably fast enough for the moment.
    val builder = ListBuffer[(Option[Span], Entity, List[Int], Double)]()
    for (i <- 0 until tokenIds.length) {
      for (j <- (i + 1) to tokenIds.length) {
        val entities = lookup(tokenIds.slice(i, j))
        for (entity <- entities) {
          builder += ((Some(Span(i, j)), entity._1, entity._2, entity._3))
        }
      }
    }
    new EntityLinking(builder.toList)
  }
}

case class Span(val start: Int, val end: Int) {

  /**
   * True if {@code index} is contained within this span.
   */
  def contains(index: Int): Boolean = {
    index >= start && index < end
  }
}