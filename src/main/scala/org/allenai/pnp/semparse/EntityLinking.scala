package org.allenai.pnp.semparse

import scala.collection.mutable.MultiMap

import com.jayantkrish.jklol.ccg.lambda.Type
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import scala.collection.mutable.ListBuffer

case class EntityLinking(map: Map[Span, Set[Entity]]) {
  
  def entities(): List[Entity] = {
    // TODO
    null
  }
}

case class Entity(val expr: Expression2, val t: Type, val template: Template, val names: List[List[Int]]) {
}

class EntityDict(val map: MultiMap[List[Int], Entity]) {
  
  def lookup(tokenIds: List[Int]): Set[Entity] = {
    if (map.contains(tokenIds)) {
      map(tokenIds).toSet
    } else {
      Set()
    }
  }
  
  def link(tokenIds: List[Int]): EntityLinking = {
    // This is a naive way to match entity names against the 
    // question text, but it's probably fast enough for the moment.
    val builder = ListBuffer[(Span, Set[Entity])]()
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