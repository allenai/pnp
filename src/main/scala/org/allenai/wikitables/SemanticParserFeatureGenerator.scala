package org.allenai.wikitables

import org.allenai.pnp.semparse.Entity
import org.allenai.pnp.semparse.Span

import scala.collection.JavaConverters._
import SemanticParserFeatureGenerator._
import edu.cmu.dynet.FloatVector
import edu.cmu.dynet.Dim
import org.apache.commons.lang3.StringUtils
import org.allenai.pnp.semparse.EntityLinking
import com.google.common.base.Preconditions
import com.jayantkrish.jklol.util.IndexedList

import scala.collection.mutable

class SemanticParserFeatureGenerator(
  featureFunctions: Array[EntityTokenFeatureFunction],
  featureFunctionsPrecomputed: Array[EntityTokenFeaturePrecomputeFunction]
)
    extends Serializable {

  val numFeatures = featureFunctions.size + featureFunctionsPrecomputed.size

  def apply(example: WikiTablesExample, entity: Entity, span: Option[Span],
    tokenToId: String => Int, table: Table): (Dim, FloatVector) = {
    // A tokens x features matrix.
    val tokens = example.sentence.getWords.asScala
    val dim = Dim(tokens.size, numFeatures)
    val matrix = new FloatVector(tokens.size * numFeatures)
    val featuresPrecomputed = for (featureFunc <- featureFunctionsPrecomputed) yield {
      featureFunc(example, entity, span, tokenToId, table)
    }
    for (i <- 0 until tokens.length) {
      for ((featureFunc, j) <- featureFunctions.zipWithIndex) {
        // Dynet stores matrices in column-major format,
        // (which in this case means that the entries for
        // each feature are consecutive).
        val index = j * tokens.size + i
        val featureValue = featureFunc(example, i, entity, span, tokenToId, table)
        matrix(index) = featureValue
      }
      for (k <- featuresPrecomputed.indices) {
        val j = k + featureFunctions.size
        val index = j * tokens.size + i
        val featureValue = featuresPrecomputed(k)(i)
        matrix(index) = featureValue
      }
    }
    (dim, matrix)
  }
}

object SemanticParserFeatureGenerator {
  type EntityTokenFeatureFunction = (WikiTablesExample, Int, Entity, Option[Span], String => Int, Table) => Float

  type EntityTokenFeaturePrecomputeFunction = (WikiTablesExample, Entity, Option[Span], String => Int, Table) => Seq[Float]

  def getWikitablesGenerator(editDistance: Boolean): SemanticParserFeatureGenerator = {
    var features: List[EntityTokenFeatureFunction] = List(
      SemanticParserFeatureGenerator.spanFeatures,
      SemanticParserFeatureGenerator.tokenExactMatchFeature,
      SemanticParserFeatureGenerator.tokenLemmaMatchFeature,
      // SemanticParserFeatureGenerator.editDistanceFeature,
      SemanticParserFeatureGenerator.relatedColumnFeature,
      SemanticParserFeatureGenerator.relatedColumnLemmaFeature
    )

    if (editDistance) {
      features = features ++ List(SemanticParserFeatureGenerator.editDistanceFeature _)
    }

    var featuresWithPrecompute: List[EntityTokenFeaturePrecomputeFunction] = List(
      SemanticParserFeatureGenerator.entityTokenSpanOverlapFeatures,
      SemanticParserFeatureGenerator.entityTokenSpanOverlapLemmaFeatures
    )

    new SemanticParserFeatureGenerator(features.toArray, featuresWithPrecompute.toArray)
  }

  def spanFeatures(ex: WikiTablesExample, tokenIndex: Int, entity: Entity,
    span: Option[Span], tokenToId: String => Int, table: Table): Float = {
    if (span.isDefined && span.get.contains(tokenIndex)) {
      1.0f
    } else {
      0.0f
    }
  }

  def tokenExactMatchFeature(ex: WikiTablesExample, tokenIndex: Int, entity: Entity,
    span: Option[Span], tokenToId: String => Int, table: Table): Float = {
    val token = ex.sentence.getWords.get(tokenIndex)
    if (entity.nameTokensSet.contains(tokenToId(token))) {
      1.0f
    } else {
      0.0f
    }
  }

  def tokenLemmaMatchFeature(ex: WikiTablesExample, tokenIndex: Int, entity: Entity,
    span: Option[Span], tokenToId: String => Int, table: Table): Float = {
    val token = ex.sentence.getWords.get(tokenIndex)
    val lemmas = ex.sentence.getAnnotation(WikiTablesUtil.LEMMA_ANNOTATION).asInstanceOf[List[String]]
    val lemma = lemmas(tokenIndex)
    if (entity.nameTokensSet.contains(tokenToId(token)) || entity.nameLemmaSet.contains(lemma)) {
      1.0f
    } else {
      0.0f
    }
  }

  def editDistanceFeature(ex: WikiTablesExample, tokenIndex: Int, entity: Entity,
    span: Option[Span], tokenToId: String => Int, table: Table): Float = {
    // Note that the returned value is <= 1.0 and can be negative if the edit distance is greater
    // than the length of the token.
    // Assuming entity string is a constant.
    val token = ex.sentence.getWords.get(tokenIndex)
    val entityString = entity.expr.toString.split("\\.").last
    1.0f - (StringUtils.getLevenshteinDistance(token, entityString).toFloat / token.length)
  }

  def relatedColumnFeature(ex: WikiTablesExample, tokenIndex: Int, entity: Entity,
    span: Option[Span], tokenToId: String => Int, table: Table): Float = {
    val token = ex.sentence.getWords.get(tokenIndex)
    if (entity.t == WikiTablesTypeDeclaration.COL_FUNCTION_TYPE) {
      val id = entity.expr.toString
      val colOpt = table.getColumn(id)
      Preconditions.checkState(colOpt.isDefined, "No such column: %s", id)
      val (col, colId) = colOpt.get
      val cells = table.cells(colId)
      val cellsWithToken = cells.filter(c => c.tokens.contains(token))

      if (cellsWithToken.size > 0) {
        1.0f
      } else {
        0.0f
      }
    } else {
      0.0f
    }
  }

  def relatedColumnLemmaFeature(ex: WikiTablesExample, tokenIndex: Int, entity: Entity,
    span: Option[Span], tokenToId: String => Int, table: Table): Float = {
    val token = ex.sentence.getWords.get(tokenIndex)
    val lemmas = ex.sentence.getAnnotation(WikiTablesUtil.LEMMA_ANNOTATION).asInstanceOf[List[String]]
    val lemma = lemmas(tokenIndex)
    if (entity.t == WikiTablesTypeDeclaration.COL_FUNCTION_TYPE) {
      val id = entity.expr.toString
      val colOpt = table.getColumn(id)
      Preconditions.checkState(colOpt.isDefined, "No such column: %s", id)
      val (col, colId) = colOpt.get
      val cells = table.cells(colId)
      val cellsWithToken = cells.filter(c => c.tokens.contains(token) || c.lemmas.contains(lemma))

      if (cellsWithToken.size > 0) {
        1.0f
      } else {
        0.0f
      }
    } else {
      0.0f
    }
  }

  // Going left to right, find maximal contiguous sequence of elements in sequence which is in
  // set and score each element according to overlap fraction. E.g.,
  // maxOverlapFractions(Seq(1,2,3,2,5,2,4), Set(1,2,3,4)) =
  //   Seq(0.75, 0.75, 0.75, 0.5, 0.0, 0.5, 0.5)
  def maxOverlapFractions[A](sequence: Seq[A], set: Set[A]): Seq[Float] = {
    val setSize = set.size.max(1)
    var stopIndex = 0
    val res = mutable.Seq.fill(sequence.size)(0f)
    for (startIndex <- sequence.indices) {
      if (stopIndex < startIndex) stopIndex = startIndex
      while (stopIndex < sequence.size && set.contains(sequence(stopIndex)) &&
        !sequence.slice(startIndex, stopIndex).contains(sequence(stopIndex))) {
        stopIndex += 1
      }
      val score = 1.0f * (stopIndex - startIndex) / setSize
      for (i <- startIndex until stopIndex) {
        res(i) = score.max(res(i))
      }
    }
    res
  }

  def maxOverlapFractionsTwoSets[A, B](
    sequence: Seq[(A, B)],
    set1: Set[A], set2: Set[B]
  ): Seq[Float] = {
    val setSize = set1.size.max(set2.size).max(1)
    var stopIndex = 0
    val res = mutable.Seq.fill(sequence.size)(0f)
    for (startIndex <- sequence.indices) {
      if (stopIndex < startIndex) stopIndex = startIndex
      while (stopIndex < sequence.size &&
        (set1.contains(sequence(stopIndex)._1) || set2.contains(sequence(stopIndex)._2)) &&
        !sequence.slice(startIndex, stopIndex).contains(sequence(stopIndex))) {
        stopIndex += 1
      }
      val score = 1.0f * (stopIndex - startIndex) / setSize
      for (i <- startIndex until stopIndex) {
        res(i) = score.max(res(i))
      }
    }
    res
  }

  def entityTokenSpanOverlapFeatures(ex: WikiTablesExample, entity: Entity,
    span: Option[Span], tokenToId: String => Int, table: Table): Seq[Float] = {
    val tokenIds = ex.sentence.getWords.asScala.map(tokenToId)
    maxOverlapFractions(tokenIds, entity.nameTokensSet)
  }

  def entityTokenSpanOverlapLemmaFeatures(ex: WikiTablesExample, entity: Entity,
    span: Option[Span], tokenToId: String => Int, table: Table): Seq[Float] = {
    val tokenIds = ex.sentence.getWords.asScala.map(tokenToId)
    val tokenLemmas =
      ex.sentence.getAnnotation(WikiTablesUtil.LEMMA_ANNOTATION).asInstanceOf[Seq[String]]
    maxOverlapFractionsTwoSets(tokenLemmas.zip(tokenIds), entity.nameLemmaSet, entity.nameTokensSet)
  }

}
