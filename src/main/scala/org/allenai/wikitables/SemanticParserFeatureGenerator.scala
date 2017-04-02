package org.allenai.wikitables

import org.allenai.pnp.semparse.Entity
import org.allenai.pnp.semparse.Span

import SemanticParserFeatureGenerator._
import edu.cmu.dynet.FloatVector
import edu.cmu.dynet.Dim
import org.allenai.pnp.semparse.EntityLinking
import com.google.common.base.Preconditions
import com.jayantkrish.jklol.util.IndexedList

class SemanticParserFeatureGenerator(featureFunctions: Array[EntityTokenFeatureFunction]) 
 extends Serializable {

  val numFeatures = featureFunctions.size
  
  def apply(tokens: Seq[String], entity: Entity, span: Option[Span], tokenToId: String => Int,
      table: Table): (Dim, FloatVector) = {
    // A tokens x features matrix.
    val dim = Dim(tokens.size, numFeatures)
    val matrix = new FloatVector(tokens.size * numFeatures)
    for ((token, i) <- tokens.zipWithIndex) {
      for ((featureFunc, j) <- featureFunctions.zipWithIndex) {
        // Dynet stores matrices in column-major format,
        // (which in this case means that the entries for
        // each feature are consecutive).
        val index = j * tokens.size + i
        val featureValue = featureFunc(token, i, entity, span, tokenToId, table)
        matrix(index) = featureValue
      }
    }
    (dim, matrix)
  }
}

object SemanticParserFeatureGenerator {
  type EntityTokenFeatureFunction = (String, Int, Entity, Option[Span], String => Int, Table) => Float

  def getWikitablesGenerator(): SemanticParserFeatureGenerator = {
    new SemanticParserFeatureGenerator(Array(
        SemanticParserFeatureGenerator.spanFeatures,
        SemanticParserFeatureGenerator.tokenExactMatchFeature,
        SemanticParserFeatureGenerator.relatedColumnFeature))
  }
  
  def spanFeatures(token: String, tokenIndex: Int, entity: Entity,
      span: Option[Span], tokenToId: String => Int, table: Table): Float = {
    if (span.isDefined && span.get.contains(tokenIndex)) {
      1.0f
    } else {
      0.0f
    }
  }

  def tokenExactMatchFeature(token: String, tokenIndex: Int, entity: Entity,
      span: Option[Span], tokenToId: String => Int, table: Table): Float = {
    if (entity.nameTokensSet.contains(tokenToId(token))) {
      1.0f
    } else {
      0.0f
    }
  }

  def relatedColumnFeature(token: String, tokenIndex: Int, entity: Entity,
      span: Option[Span], tokenToId: String => Int, table: Table): Float = {
    if (entity.t == WikiTablesTypeDeclaration.COL_FUNCTION_TYPE) {
      val id = entity.expr.toString
      val colOpt = table.getColumn(id)
      Preconditions.checkState(colOpt.isDefined, "No such column: %s", id)
      val (col, colId) = colOpt.get
      val cells = table.cells(colId)
      val cellsWithToken = cells.filter(c => c.idTokens.contains(token))

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
