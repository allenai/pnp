package org.allenai.wikitables

import org.allenai.pnp.semparse.Entity
import org.allenai.pnp.semparse.Span

import EntityTokenFeatureGenerator._
import edu.cmu.dynet.FloatVector
import edu.cmu.dynet.Dim

class EntityTokenFeatureGenerator(featureFunctions: Array[EntityTokenFeatureFunction]) {

  val numFeatures = featureFunctions.size
  
  def apply(tokens: Seq[Int], entity: Entity, span: Option[Span]): (Dim, FloatVector) = {
    // A tokens x features matrix.
    val dim = Dim(tokens.size, numFeatures)
    val matrix = new FloatVector(tokens.size * numFeatures)
    for ((token, i) <- tokens.zipWithIndex) {
      for ((featureFunc, j) <- featureFunctions.zipWithIndex) {
        // Dynet stores matrices in column-major format,
        // (which in this case means that the entries for
        // each feature are consecutive).
        val index = j * tokens.size + i
            
        val featureValue = featureFunc(token, i, entity, span)
        matrix(index) = featureValue
      }
    }
    (dim, matrix)
  }
}

object EntityTokenFeatureGenerator {
  type EntityTokenFeatureFunction = (Int, Int, Entity, Option[Span]) => Float

  def getWikitablesGenerator(): EntityTokenFeatureGenerator = {
    new EntityTokenFeatureGenerator(Array(
        EntityTokenFeatureGenerator.spanFeatures,
        EntityTokenFeatureGenerator.tokenExactMatchFeature))
  }
  
  def spanFeatures(tokenId: Int, tokenIndex: Int, entity: Entity,
      span: Option[Span]): Float = {
    if (span.isDefined && span.get.contains(tokenIndex)) {
      1.0f
    } else {
      0.0f
    }
  }

  def tokenExactMatchFeature(tokenId: Int, tokenIndex: Int, entity: Entity,
      span: Option[Span]): Float = {
    if (entity.nameTokensSet.contains(tokenId)) {
      1.0f
    } else {
      0.0f
    }
  }
}
