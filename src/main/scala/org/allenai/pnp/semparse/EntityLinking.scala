package org.allenai.pnp.semparse

import scala.collection.mutable.MultiMap

import com.jayantkrish.jklol.ccg.lambda.Type
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import scala.collection.mutable.ListBuffer

case class EntityLinking(map: Map[Span, Set[(Entity, List[Int], Double)]]) {
  val entities = map.values.flatten.map(_._1).toSet.toList
  
  val entityMatches = SemanticParser.seqToMultimap(map.values
      .flatten.map(x => (x._1, (x._2, x._3))).toSeq)
  val bestEntityMatches = entityMatches.map(x => (x._1, x._2.maxBy(_._2)))
  val bestEntityMatchesList = bestEntityMatches.map(x => (x._1, x._2._1, x._2._2)).toList 
  
  def getEntitiesWithType(t: Type): List[Entity] = {
    entities.filter(_.t.equals(t))
  }
}

case class Entity(val expr: Expression2, val t: Type,
    val template: Template, val names: List[List[Int]]) {
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
    val builder = ListBuffer[(Span, Set[(Entity, List[Int], Double)])]()
    for (i <- 0 until tokenIds.length) {
      for (j <- (i + 1) to tokenIds.length) {
        val entities = lookup(tokenIds.slice(i, j))
        if (entities.size > 0) {
          builder += ((Span(i, j), entities))
        }
      }
    }
    new EntityLinking(builder.toMap)
  }
}

case class Span(val start: Int, val end: Int) 