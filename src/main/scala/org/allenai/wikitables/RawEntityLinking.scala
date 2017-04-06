package org.allenai.wikitables

import scala.collection.JavaConverters._

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Paths

import scala.collection.mutable.ListBuffer
import scala.io.Source

import org.allenai.pnp.semparse.ConstantTemplate
import org.allenai.pnp.semparse.Entity
import org.allenai.pnp.semparse.EntityLinking
import org.allenai.pnp.semparse.SemanticParserUtils
import org.allenai.pnp.semparse.Span

import com.google.common.base.Preconditions
import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import com.jayantkrish.jklol.ccg.lambda2.StaticAnalysis

import edu.cmu.dynet._
import edu.stanford.nlp.sempre.Formula
import spray.json.pimpAny
import spray.json.pimpString
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration
import org.allenai.pnp.semparse.ApplicationTemplate
import com.jayantkrish.jklol.ccg.lambda2.Expression2

/**
 * A raw entity linking. These linkings are mostly generated 
 * by Sempre, and need to be preprocessed before they can be
 * used in the semantic parser.
 */
case class RawEntityLinking(id: String, links: List[(Option[Span], Formula)]) {
  
  import RawEntityLinking._
  
  def toEntityLinking(ex: WikiTablesExample, tokenToId: String => Int,
    featureGenerator: SemanticParserFeatureGenerator, table: Table,
    typeDeclaration: TypeDeclaration): EntityLinking = {

    val builder = ListBuffer[(Entity, Dim, FloatVector)]()
    for (link <- links) {
      val entityString = link._2.toString
      val entityExpr = ExpressionParser.expression2().parse(entityString)

      // The entity linking may contain whole logical forms, which at
      // the moment are restricted to the form (or entity1 entity2).
      // These are problematic because the parser can generate that expression
      // multiple ways. Filter them out for now.
      if (entityExpr.isConstant()) {
        val entityType = StaticAnalysis.inferType(entityExpr, typeDeclaration)
        Preconditions.checkState(!SemanticParserUtils.isBadType(entityType),
            "Found bad type %s for expression %s", entityType, entityExpr)

        val template = if (entityType == Seq2SeqTypeDeclaration.seqType) {
          // Hack to handle seq2seq models
          ApplicationTemplate(Seq2SeqTypeDeclaration.endType,
              // Second argument doesn't matter as it gets replaced
              // during parsing.
              Expression2.nested(List(entityExpr, holeExpr).asJava),
              List((2, Seq2SeqTypeDeclaration.endType, false)))
        } else {
          ConstantTemplate(entityType, entityExpr)
        }

        val span = link._1

        // Tokens in the names of entities are also encoded with the
        // example-specific vocabulary.
        val entityTokens = table.tokenizeEntity(entityString)
        val entityTokenIds = entityTokens.map(tokenToId(_)).toList
        val entityLemmas = table.lemmatizeEntity(entityString)
        val entity = Entity(entityExpr, template.root, template, List(entityTokenIds),
            List(entityLemmas))

        // Generate entity/token features.
        val (dim, featureMatrix) = featureGenerator.apply(ex, entity, span, tokenToId, table)
        
        builder += ((entity, dim, featureMatrix))
      }
    }
    
    val entities = builder.map(_._1).toArray
    val features = builder.map(x => (x._2, x._3)).toArray

    new EntityLinking(entities, features)
  }
}

object RawEntityLinking {
  import WikiTablesJsonFormat._
 
  val holeExpr = Expression2.constant("hole")
  
  def fromJsonFile(filename: String): Seq[RawEntityLinking] = {
    val content = Source.fromFile(filename).getLines.mkString(" ")
    content.parseJson.convertTo[List[RawEntityLinking]]
  }

  def toJsonFile(filename: String, linkings: Seq[RawEntityLinking]): Unit = {
    val json = linkings.toArray.toJson
    Files.write(Paths.get(filename), json.prettyPrint.getBytes(StandardCharsets.UTF_8))
  }
}
