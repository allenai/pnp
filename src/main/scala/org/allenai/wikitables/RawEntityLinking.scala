package org.allenai.wikitables

import edu.stanford.nlp.sempre.Formula
import org.allenai.pnp.semparse.Span
import org.allenai.pnp.semparse.EntityLinking
import org.allenai.pnp.semparse.ConstantTemplate
import com.jayantkrish.jklol.ccg.lambda2.StaticAnalysis
import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import org.allenai.pnp.semparse.Entity
import org.allenai.pnp.semparse.SemanticParserUtils
import com.google.common.base.Preconditions
import scala.collection.mutable.ListBuffer
import spray.json._
import scala.io.Source
import java.nio.file.Paths
import java.nio.file.Files
import java.nio.charset.StandardCharsets

/**
 * A raw entity linking. These linkings are mostly generated 
 * by Sempre, and need to be preprocessed before they can be
 * used in the semantic parser.
 */
case class RawEntityLinking(id: String, links: List[(Option[Span], Formula)]) {

  def toEntityLinking(tokenToId: String => Int,
    typeDeclaration: WikiTablesTypeDeclaration): EntityLinking = {

    val builder = ListBuffer[(Option[Span], Entity, List[Int], Double)]()
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

        val template = ConstantTemplate(entityType, entityExpr)

        val span = link._1

        // Tokens in the names of entities are also encoded with the
        // example-specific vocabulary.
        val entityTokens = WikiTablesUtil.tokenizeEntity(entityString)
        val entityTokenIds = entityTokens.map(tokenToId(_)).toList
        val entity = Entity(entityExpr, entityType, template, List(entityTokenIds))
        builder += ((span, entity, entityTokenIds, 0.1))
      }
    }
    new EntityLinking(builder.toList)
  }
}

object RawEntityLinking {
  import WikiTablesJsonFormat._
  
  def fromJsonFile(filename: String): Seq[RawEntityLinking] = {
    val content = Source.fromFile(filename).getLines.mkString(" ")
    content.parseJson.convertTo[List[RawEntityLinking]]
  }

  def toJsonFile(filename: String, linkings: Seq[RawEntityLinking]): Unit = {
    val json = linkings.toArray.toJson
    Files.write(Paths.get(filename), json.prettyPrint.getBytes(StandardCharsets.UTF_8))
  }
}
