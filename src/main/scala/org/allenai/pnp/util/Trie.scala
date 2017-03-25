package org.allenai.pnp.util

import scala.collection.mutable.{ Map => MutableMap, Set => MutableSet }

class Trie[T] {
  
  val next = MutableMap[Int, MutableMap[T, Int]]()
  val startNodeId = 0
  var numNodes = 0
  val inTrie = MutableSet[Int]()

  /**
   * Insert a key into this trie.
   */
  def insert(key: Seq[T]): Unit = {
    if (!next.contains(startNodeId)) {
      next.put(startNodeId, MutableMap[T, Int]())
      numNodes += 1
    }
    
    insertHelper(key, startNodeId)
  }
  
  private def insertHelper(key: Seq[T], currentNodeId: Int): Unit = {
    if (key.size == 0) {
      inTrie.add(currentNodeId)
    } else {
      if (!next(currentNodeId).contains(key.head)) {
        val nextNodeId = numNodes
        numNodes += 1
        next.put(nextNodeId, MutableMap[T, Int]())
        next(currentNodeId).put(key.head, nextNodeId)
      }

      val nextNodeId = next(currentNodeId)(key.head)
      insertHelper(key.tail, nextNodeId)
    }
  }
  
  /**
   * Lookup a key prefix in this trie. If the trie contains
   * a key with that prefix, returns the id of the trie node 
   * corresponding to that prefix.
   */
  def lookup(keyPrefix: Seq[T]): Option[Int] = {
    if (numNodes > 0) {
      lookup(keyPrefix, startNodeId)
    } else {
      None
    }
  }

  def lookup(keyPrefix: Seq[T], currentNodeId: Int): Option[Int] = {
    if (keyPrefix.size == 0) {
      Some(currentNodeId)
    } else {
      val nextEdges = next(currentNodeId)
      val nextNodeId = nextEdges.get(keyPrefix.head)
      if (nextNodeId.isDefined) {
        lookup(keyPrefix.tail, nextNodeId.get)
      } else {
        None
      }
    }
  }

  /**
   * Gets the map to next trie nodes for a current node.  
   */
  def getNextMap(nodeId: Int): MutableMap[T, Int] = {
    next(nodeId)
  }
}