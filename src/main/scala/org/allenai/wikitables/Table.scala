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

case class Table(id: String, columns: List[Column], cells: List[List[Cell]])
case class Column(id: String, originalString: String)
case class Cell(id: String, originalString: String)

object Table {
  
  val PREPROCESSING_SUFFIX = ".tables.json"
  
  import WikiTablesJsonFormat._
  
  def knowledgeGraphToTable(graph: TableKnowledgeGraph): Table = {
    val columns = graph.columns.asScala.map(c =>
      Column(c.relationNameValue.id, c.originalString)).toList
    val cells = graph.columns.asScala.map(c => 
      c.children.asScala.map(x => Cell(x.properties.id, x.properties.originalString)).toList
    ).toList
    
    Table(graph.filename, columns, cells)
  }
  
  def fromJsonFile(filename: String): Seq[Table] = {
    val content = Source.fromFile(filename).getLines.mkString(" ")
    content.parseJson.convertTo[List[Table]]
  }

  def toJsonFile(filename: String, tables: Iterable[Table]): Unit = {
    val json = tables.toArray.toJson
    Files.write(Paths.get(filename), json.prettyPrint.getBytes(StandardCharsets.UTF_8))
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
        knowledgeGraphToTable(graph)
      }
      
      Table.toJsonFile(preprocessedFile, tables)
      tables
    }

    tables.toVector
  }
}
