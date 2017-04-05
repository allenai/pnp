package org.allenai.pnp.semparse

import scala.collection.JavaConverters._
import scala.collection.mutable.MultiMap

import com.jayantkrish.jklol.ccg.lambda.Type
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import scala.collection.mutable.ListBuffer
import edu.cmu.dynet.FloatVector
import edu.cmu.dynet.Dim
import com.google.common.base.Preconditions
import com.jayantkrish.jklol.util.IndexedList

case class EntityLinking(entities: Array[Entity],
    entityTokenFeatures: Array[(Dim, FloatVector)]) {
  Preconditions.checkArgument(entities.length == entityTokenFeatures.length)
  
  val entityIndex = IndexedList.create(entities.toList.asJava)
  val entityTypes = entities.map(_.t).toSet
  val entitiesWithType = entityTypes.map(t => (t, entities.filter(_.t.equals(t)).toArray)).toMap
  
  def getEntitiesWithType(t: Type): Array[Entity] = {
    entitiesWithType.getOrElse(t, Array())
  }
  
  def getTokenFeatures(entity: Entity): (Dim, FloatVector) = {
    val index = entityIndex.getIndex(entity)
    entityTokenFeatures(index)
  }
}

case class Entity(val expr: Expression2, val t: Type,
    val template: Template, val names: List[List[Int]],
    val nameLemmas: List[List[String]]) {

  val nameTokens = names.flatten.toArray
  val nameTokensSet = names.flatten.toSet
  val nameLemmaSet = nameLemmas.flatten.toSet
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
    val builder = ListBuffer[(Span, Entity)]()
    for (i <- 0 until tokenIds.length) {
      for (j <- (i + 1) to tokenIds.length) {
        val entities = lookup(tokenIds.slice(i, j))
        for (entity <- entities) {
          builder += ((Span(i, j), entity))
        }
      }
    }

    // TODO: spans to features here.
    val entityTokenFeatures: Array[(Dim, FloatVector)] = null
    new EntityLinking(builder.map(x => x._2).toSet.toArray, entityTokenFeatures)
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