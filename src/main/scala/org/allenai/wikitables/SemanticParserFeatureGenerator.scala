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
  featureFunctionsPrecomputed: Array[EntityTokenFeaturePrecomputeFunction],
  val vocab: IndexedList[String],
  val vocabCounts: Array[Double]
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
      featureFunc(example, entity, span, tokenToId, table, this)
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

  // Lower weight on very frequent words
  def getWordWeight(wordId: Int): Float = {
    val count = if (wordId < 0 || wordId >= vocabCounts.size) 1d else vocabCounts(wordId)
    1.0f / math.log(count + WORD_WEIGHT_SMOOTHING).toFloat
  }

  // For known words check if they're alphanumeric, otherwise assume true
  def isAlphaNumeric(wordId: Int): Boolean = {
    if (wordId < vocab.size) {
      isAlphaNumeric(vocab.get(wordId))
    } else {
      true
    }
  }

  def isAlphaNumeric(word: String): Boolean = {
    StringUtils.isAlphanumeric(word)
  }

}

object SemanticParserFeatureGenerator {
  type EntityTokenFeatureFunction = (WikiTablesExample, Int, Entity, Option[Span], String => Int, Table) => Float

  type EntityTokenFeaturePrecomputeFunction =
    (WikiTablesExample, Entity, Option[Span], String => Int, Table, SemanticParserFeatureGenerator) => Seq[Float]

  val WORD_WEIGHT_SMOOTHING = 5

  def getWikitablesGenerator(editDistance: Boolean, vocab: IndexedList[String],
      vocabCounts: Array[Double]): SemanticParserFeatureGenerator = {
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

    new SemanticParserFeatureGenerator(features.toArray,
      featuresWithPrecompute.toArray, vocab, vocabCounts)
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
  // maxOverlapFractions(Seq(1,2,3,2,5,2,4), Set(1,2,3,4), (x:Int) => 1f) =
  //   Seq(0.75, 0.75, 0.75, 0.5, 0.0, 0.5, 0.5)
  def maxOverlapFractions[A](sequence: Seq[A], set: Set[A], scorer: A => Float): Seq[Float] = {
    val setScore = if (set.isEmpty) {
      1.0f
    } else {
      set.toSeq.map(scorer).sum
    }
    val sequenceScores = sequence.map(scorer)
    var stopIndex = 0
    val res = mutable.Seq.fill(sequence.size)(0f)
    for (startIndex <- sequence.indices) {
      if (stopIndex < startIndex) stopIndex = startIndex
      while (stopIndex < sequence.size && set.contains(sequence(stopIndex)) &&
        !sequence.slice(startIndex, stopIndex).contains(sequence(stopIndex))) {
        stopIndex += 1
      }
      val score = sequenceScores.slice(startIndex, stopIndex).sum / setScore
      for (i <- startIndex until stopIndex) {
        res(i) = score.max(res(i))
      }
    }
    res
  }

  // Like maxOverlapFractions, but there is a secondary set also used for matching, but not scoring
  def maxOverlapFractionsTwoSets[A, B](
    sequence: Seq[(A, B)], set1: Set[A], set2: Set[B], scorer: A => Float
  ): Seq[Float] = {
    val setScore = if (set1.isEmpty) {
      1.0f
    } else {
      set1.toSeq.map(scorer).sum
    }
    val sequenceScores = sequence.map(x => scorer(x._1))
    var stopIndex = 0
    val res = mutable.Seq.fill(sequence.size)(0f)
    for (startIndex <- sequence.indices) {
      if (stopIndex < startIndex) stopIndex = startIndex
      while (stopIndex < sequence.size &&
        (set1.contains(sequence(stopIndex)._1) || set2.contains(sequence(stopIndex)._2)) &&
        !sequence.slice(startIndex, stopIndex).contains(sequence(stopIndex))) {
        stopIndex += 1
      }
      val score = sequenceScores.slice(startIndex, stopIndex).sum / setScore
      for (i <- startIndex until stopIndex) {
        res(i) = score.max(res(i))
      }
    }
    res
  }

  def entityTokenSpanOverlapFeatures(ex: WikiTablesExample, entity: Entity,
    span: Option[Span], tokenToId: String => Int, table: Table,
    featureGenerator: SemanticParserFeatureGenerator): Seq[Float] = {
    val tokenIds = ex.sentence.getWords.asScala.map(tokenToId)
    val entityNameTokenSet = entity.nameTokensSet.filter(featureGenerator.isAlphaNumeric)
    maxOverlapFractions(tokenIds, entityNameTokenSet, featureGenerator.getWordWeight)
  }

  def entityTokenSpanOverlapLemmaFeatures(ex: WikiTablesExample, entity: Entity,
    span: Option[Span], tokenToId: String => Int, table: Table,
    featureGenerator: SemanticParserFeatureGenerator): Seq[Float] = {
    val tokenIds = ex.sentence.getWords.asScala.map(tokenToId)
    val tokenLemmas =
      ex.sentence.getAnnotation(WikiTablesUtil.LEMMA_ANNOTATION).asInstanceOf[Seq[String]]
    val entityNameTokenSet = entity.nameTokensSet.filter(featureGenerator.isAlphaNumeric)
    val entityNameLemmaSet = entity.nameLemmaSet.filter(featureGenerator.isAlphaNumeric)
    maxOverlapFractionsTwoSets(tokenIds.zip(tokenLemmas), entityNameTokenSet,
      entity.nameLemmaSet, featureGenerator.getWordWeight)
  }

}
