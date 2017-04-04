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

class SemanticParserFeatureGenerator(featureFunctions: Array[EntityTokenFeatureFunction]) 
 extends Serializable {

  val numFeatures = featureFunctions.size
  
  def apply(example: WikiTablesExample, entity: Entity, span: Option[Span],
      tokenToId: String => Int, table: Table): (Dim, FloatVector) = {
    // A tokens x features matrix.
    val tokens = example.sentence.getWords.asScala
    val dim = Dim(tokens.size, numFeatures)
    val matrix = new FloatVector(tokens.size * numFeatures)
    for (i <- 0 until tokens.length) {
      for ((featureFunc, j) <- featureFunctions.zipWithIndex) {
        // Dynet stores matrices in column-major format,
        // (which in this case means that the entries for
        // each feature are consecutive).
        val index = j * tokens.size + i
        val featureValue = featureFunc(example, i, entity, span, tokenToId, table)
        matrix(index) = featureValue
      }
    }
    (dim, matrix)
  }
}

object SemanticParserFeatureGenerator {
  type EntityTokenFeatureFunction = (WikiTablesExample, Int, Entity, Option[Span], String => Int, Table) => Float

  def getWikitablesGenerator(): SemanticParserFeatureGenerator = {
    new SemanticParserFeatureGenerator(Array(
        SemanticParserFeatureGenerator.spanFeatures,
        SemanticParserFeatureGenerator.tokenExactMatchFeature,
        SemanticParserFeatureGenerator.tokenLemmaMatchFeature,
        // SemanticParserFeatureGenerator.editDistanceFeature,
        SemanticParserFeatureGenerator.relatedColumnFeature,
        SemanticParserFeatureGenerator.relatedColumnLemmaFeature))
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
}
