package org.allenai.wikitables

import java.nio.file.Files
import java.nio.file.Paths

import scala.collection.JavaConverters._
import scala.io.Source
import com.google.common.base.Preconditions
import edu.stanford.nlp.sempre.ContextValue
import edu.stanford.nlp.sempre.tables.TableKnowledgeGraph
import fig.basic.LispTree
import spray.json._
import java.nio.charset.StandardCharsets

import edu.stanford.nlp.sempre.LanguageAnalyzer
import org.allenai.pnp.semparse.KnowledgeGraph

case class Table(id: String, columns: Array[Column], cells: Array[Array[Cell]]) {

  val colIdMap = columns.zipWithIndex.map(x => (x._1.id, (x._1, x._2))).toMap
  val cellIdMap = cells.flatMap(cs => cs.map(c => (c.id, c))).toMap

  def getColumn(colId: String): Option[(Column, Int)] = {
    colIdMap.get(colId)
  }

  def getCell(cellId: String): Option[Cell] = {
    cellIdMap.get(cellId)
  }

  def toKnowledgeGraph(): KnowledgeGraph = {
    KnowledgeGraph(cells.flatten.map(cell => (cell.id, cell.columnId)).toList)
  }

  /**
   * Converts the id of an entity to a sequence of
   * tokens in its name. There are two cases: (1) if the entity
   * corresponds to a column or cell of this table, the
   * tokens are taken from the language analyzer's tokenization;
   * (2) otherwise, the id is split into tokens using 
   * a heuristic segmentation.
   */
  def tokenizeEntity(entityId: String): List[String] = {
    val col = getColumn(entityId)
    val cell = getCell(entityId)    
    if (col.isDefined) {
      col.get._1.tokens.toList
    } else if (cell.isDefined) {
      cell.get.tokens.toList
    } else {
      val lastDotInd = entityId.lastIndexOf(".")
      val entityTokens = if (lastDotInd == -1) {
        entityId.split('_').toList
      } else {
        val (typeName, entityName) = entityId.splitAt(lastDotInd)
        val tokens = entityName.substring(1).split('_').toList
        // Return the type prefixed to the tokens
        // List(typeName) ++ tokens

        // Return only the tokens
        tokens
      }

      // println("tokens: " + entityId + " " + entityTokens)
      entityTokens
    }
  }

  def lemmatizeEntity(entityId: String): List[String] = {
    val col = getColumn(entityId)
    val cell = getCell(entityId)    
    if (col.isDefined) {
      col.get._1.lemmas.toList
    } else if (cell.isDefined) {
      cell.get.lemmas.toList
    } else {
      List()
    }
  }

  def toTableJson = TableJson(id, columns, cells)
}

case class TableJson(id: String, columns: Array[Column], cells: Array[Array[Cell]])
case class Column(id: String, originalString: String, tokens: Array[String],
                  pos: Array[String], ner: Array[NerTag], lemmas: Array[String])
case class Cell(id: String, originalString: String, tokens: Array[String],
    pos: Array[String], ner: Array[NerTag], lemmas: Array[String], columnId: String)

case class NerTag(tag: String, value: Option[String])

object Table {
  
  val PREPROCESSING_SUFFIX = ".tables.json"
  
  import WikiTablesJsonFormat._
  
  def knowledgeGraphToTable(id: String, graph: TableKnowledgeGraph): Table = {
    val analyzer = LanguageAnalyzer.getSingleton()
    
    val columns = graph.columns.asScala.map { c =>
      val info = analyzer.analyze(c.originalString)
      val tokens = info.tokens.asScala.toArray
      val pos = info.posTags.asScala.toArray
      val nerValues = info.nerValues.asScala.map{ x =>  
        if (x == null) { None } else { Some(x) }
      }
      val ner = info.nerTags.asScala.zip(nerValues).map(x => 
        NerTag(x._1, x._2)).toArray
      val lemmas = info.lemmaTokens.asScala.toArray
      Column(c.relationNameValue.id, c.originalString, tokens, pos, ner, lemmas)
    }

    val cells = graph.columns.asScala.map{ c => 
      c.children.asScala.map{ x =>
        val info = analyzer.analyze(x.properties.originalString)
        val tokens = info.tokens.asScala.toArray
        val pos = info.posTags.asScala.toArray
        val nerValues = info.nerValues.asScala.map{ y =>  
          if (y == null) { None } else { Some(y) }
        }
        val ner = info.nerTags.asScala.zip(nerValues).map(y => 
          NerTag(y._1, y._2)).toArray
        val lemmas = info.lemmaTokens.asScala.map(x => x.toLowerCase()).toArray
        val columnId = c.relationNameValue.id
        Cell(x.properties.id, x.properties.originalString, tokens, pos, ner, lemmas, columnId)
        //Cell(x.properties.id, x.properties.originalString, tokens, pos, ner, lemmas)
      }.toArray
    }
    
    Table(id, columns.toArray, cells.toArray)
  }

  def fromJsonFile(filename: String): Seq[Table] = {
    val content = Source.fromFile(filename).getLines.mkString(" ")
    content.parseJson.convertTo[List[Table]]
  }

  def toJsonFile(filename: String, tables: Iterable[Table]): Unit = {
    val json = tables.toArray.toJson
    Files.write(Paths.get(filename), json.prettyPrint.getBytes(StandardCharsets.UTF_8))
  }

  def fromTableJson(tj: TableJson) = {
    Table(tj.id, tj.columns, tj.cells)
  }
  
  def loadDataset(filename: String, examples: Seq[WikiTablesExample]): Vector[Table] = {
    val preprocessedFile = filename + PREPROCESSING_SUFFIX

    val tables = if (Files.exists(Paths.get(preprocessedFile))) {
      Table.fromJsonFile(preprocessedFile)
    } else {
      val tableStrings = examples.map(_.tableString).toSet  
      val tables = tableStrings.map { tableString => 
        val contextValue = new ContextValue(LispTree.proto.parseFromString(tableString))
        val graph = contextValue.graph.asInstanceOf[TableKnowledgeGraph]
        knowledgeGraphToTable(tableString, graph)
      }
      
      Table.toJsonFile(preprocessedFile, tables)
      tables
    }

    tables.toVector
  }
}
