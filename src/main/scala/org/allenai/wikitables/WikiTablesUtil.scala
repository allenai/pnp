package org.allenai.wikitables

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Paths
import java.util.LinkedList
import java.util.regex.Pattern

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.io.Source

import org.allenai.pnp.semparse.ConstantTemplate
import org.allenai.pnp.semparse.Entity
import org.allenai.pnp.semparse.EntityLinking
import org.allenai.pnp.semparse.SemanticParserUtils
import org.allenai.pnp.semparse.Span
import org.json4s.DefaultFormats
import org.json4s.JNothing
import org.json4s.JValue
import org.json4s.JsonDSL.jobject2assoc
import org.json4s.JsonDSL.pair2Assoc
import org.json4s.JsonDSL.pair2jvalue
import org.json4s.JsonDSL.seq2jvalue
import org.json4s.JsonDSL.string2jvalue
import org.json4s.jvalue2extractable
import org.json4s.jvalue2monadic
import org.json4s.native.JsonMethods.parse
import org.json4s.native.JsonMethods.pretty
import org.json4s.native.JsonMethods.render
import org.json4s.string2JsonInput

import com.google.common.base.Preconditions
import com.google.common.collect.Lists
import com.jayantkrish.jklol.ccg.CcgExample
import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.ccg.lambda2.ExpressionSimplifier
import com.jayantkrish.jklol.ccg.lambda2.StaticAnalysis
import com.jayantkrish.jklol.ccg.lambda2.VariableCanonicalizationReplacementRule
import com.jayantkrish.jklol.nlpannotation.AnnotatedSentence
import com.jayantkrish.jklol.util.CountAccumulator
import com.jayantkrish.jklol.util.IndexedList

import edu.stanford.nlp.sempre.Formula
import edu.stanford.nlp.sempre.Formulas
import edu.stanford.nlp.sempre.LambdaFormula
import edu.stanford.nlp.sempre.LanguageAnalyzer
import edu.stanford.nlp.sempre.Values
import edu.stanford.nlp.sempre.tables.TableKnowledgeGraph
import edu.stanford.nlp.sempre.tables.test.CustomExample
import fig.basic.LispTree
import fig.basic.Pair
import scala.collection.mutable.ListBuffer

/**
 * This object has two main functions: (1) loading and preprocessing data (including functionality
 * around computing vocabulary, and worrying about token ids), and (2) converting between Sempre
 * data structures and PNP data structures.
 */
object WikiTablesUtil {
  implicit val formats = DefaultFormats
  val UNK = "<UNK>"
  val ENTITY = "<ENTITY>"
  val preprocessingSuffix = ".preprocessed.json"
  val simplifier = ExpressionSimplifier.lambdaCalculus()
  
  // Names of annotations
  val NER_ANNOTATION = "NER"
  val LEMMA_ANNOTATION = "lemma"

  // Maximum number of derivations stored
  val MAX_DERIVATIONS = 200

  CustomExample.opts.allowNoAnnotation = true
  TableKnowledgeGraph.opts.baseCSVDir = "data/WikiTableQuestions"
  LanguageAnalyzer.opts.languageAnalyzer = "corenlp.CoreNLPAnalyzer"

  def readDatasetFromJson(filename: String): Seq[WikiTablesExample] = {
    val fileContents = Source.fromFile(filename).getLines.mkString(" ")
    val json = parse(fileContents)
    json.children.map(exampleFromJson)
  }

  def saveDatasetToJson(dataset: Seq[WikiTablesExample], filename: String) {
    val json: JValue = dataset.map(exampleToJson)
    Files.write(Paths.get(filename), pretty(render(json)).getBytes(StandardCharsets.UTF_8))
  }

  def exampleToJson(example: WikiTablesExample): JValue = {
    val goldLogicalFormJson = example.goldLogicalForm match {
      case None => JNothing
      case Some(lf) => ("gold logical form" -> WikiTablesUtil.toSempreLogicalForm(lf).toString): JValue
    }
    goldLogicalFormJson merge
    ("id" -> example.id) ~
      ("question" -> example.sentence.getWords.asScala.mkString(" ")) ~
      ("tokens" -> example.sentence.getWords.asScala) ~
      ("posTags" -> example.sentence.getPosTags.asScala) ~
      (NER_ANNOTATION -> example.sentence.getAnnotation(NER_ANNOTATION).asInstanceOf[Seq[Seq[String]]]) ~
      (LEMMA_ANNOTATION -> example.sentence.getAnnotation(LEMMA_ANNOTATION).asInstanceOf[Seq[String]]) ~
      ("table" -> example.tableString) ~
      ("answer" -> example.targetValue.toLispTree.toString) ~
      ("possible logical forms" -> example.possibleLogicalForms.map(WikiTablesUtil.toSempreLogicalForm).map(_.toString).toList)
  }

  def exampleFromJson(json: JValue): WikiTablesExample = {
    // Just reading the JSON here.
    val id = (json \ "id").extract[String]
    val question = (json \ "question").extract[String]
    val tokens = (json \ "tokens").extract[List[String]]
    val posTags = (json \ "posTags").extract[List[String]]
    val ner = (json \ NER_ANNOTATION).extract[List[List[String]]]
    val lemmas = (json \ LEMMA_ANNOTATION).extract[List[String]]
    val tableString = (json \ "table").extract[String]
    val answerString = (json \ "answer").extract[String]
    val goldLogicalFormString = (json \ "gold logical form") match {
      case JNothing => None
      case jval => Some(jval.extract[String])
    }
    val possibleLogicalFormStrings = (json \ "possible logical forms").extract[List[String]]

    // Now we actually construct the objects.
    val goldLogicalForm = goldLogicalFormString
      .map(Formula.fromString).map(WikiTablesUtil.toPnpLogicalForm).map(simplifier.apply)
    val possibleLogicalForms = possibleLogicalFormStrings
      .map(Formula.fromString).map(WikiTablesUtil.toPnpLogicalForm).map(simplifier.apply).toSet
    val sentence = new AnnotatedSentence(tokens.asJava, posTags.asJava, new java.util.HashMap())
    sentence.getAnnotations().put(NER_ANNOTATION, ner)
    sentence.getAnnotations().put(LEMMA_ANNOTATION, lemmas)
    new WikiTablesExample(
      id,
      sentence,
      goldLogicalForm,
      possibleLogicalForms,
      tableString,
      Values.fromLispTree(LispTree.proto.parseFromString(answerString))
    )
  }

  def convertCustomExampleToWikiTablesExample(example: CustomExample): WikiTablesExample = {
    // First we worry about the sentence; tokens, pos tags, NER, etc.
    val sentence = new AnnotatedSentence(
      example.getTokens(),
      example.languageInfo.posTags,
      new java.util.HashMap[String, Object]  // need to pass this so we can modify it later
    )
    val ner = example.languageInfo.nerTags.asScala.zip(example.languageInfo.nerValues.asScala)
    val filteredNer = ner.map { case (tag, label) => {
      if (tag == "O" && label == null) List() else List(tag, label)
    }}
    sentence.getAnnotations().put(NER_ANNOTATION, filteredNer.toList)
    
    val lemmas = example.languageInfo.lemmaTokens.asScala
    sentence.getAnnotations().put(LEMMA_ANNOTATION, lemmas.toList)

    // Then we worry about the logical forms.
    val goldLogicalForm = if (example.targetFormula == null) {
      None
    } else {
      Some(simplifier.apply(WikiTablesUtil.toPnpLogicalForm(example.targetFormula)))
    }
    val possibleLogicalForms = example.alternativeFormulas.asScala
      .map(WikiTablesUtil.toPnpLogicalForm)
      .map(simplifier.apply)
      .toSet

    // Last, we get the target value and the context (as a string).
    WikiTablesExample(
      example.id,
      sentence,
      goldLogicalForm,
      possibleLogicalForms,
      example.context.toLispTree().toString(),
      example.targetValue
    )
  }

  def getTokenCounts(examples: Iterable[WikiTablesExample]): CountAccumulator[String] = {
    val acc = CountAccumulator.create[String]
    for (example <- examples) {
      example.sentence.getWords.asScala.map(x => acc.increment(x, 1.0))
    }
    acc
  }

  def getEntityTokenCounts(data: Iterable[RawExample]): CountAccumulator[String] = {
    val acc = CountAccumulator.create[String]
    for (d <- data) {
      for (cellId <- d.table.cellIdMap.keys) {
        for (token <- d.table.tokenizeEntity(cellId)) {
          acc.increment(token, 1.0)
        }
      }
      for (colId <- d.table.colIdMap.keys) {
        for (token <- d.table.tokenizeEntity(colId)) {
          acc.increment(token, 1.0)
        }
      }
    }
    acc
  }

  def computeVocabulary(trainingDataWithEntities: Seq[RawExample], threshold: Int) = {
    val wordCounts = getTokenCounts(trainingDataWithEntities.map(_.ex))
    val entityCounts = getEntityTokenCounts(trainingDataWithEntities)
    // Vocab consists of all words that appear more than once in
    // the training data.
    val vocab = IndexedList.create(wordCounts.getKeysAboveCountThreshold(threshold + 0.1))
    // This line adds in words that appear in the names of entities.
    // Adding these to the vocabulary seems to cause massive overfitting.
    // vocab.addAll(IndexedList.create(entityCounts.getKeysAboveCountThreshold(0.0)))
    vocab.add(UNK)
    vocab.add(ENTITY)
    println(vocab.size + " words")

    for (w <- vocab.items().asScala.sorted) {
      println("  " + w + " " + wordCounts.getCount(w) + " " + entityCounts.getCount(w))
    }
    vocab
  }

  /** Returns a modified dataset that has CcgExamples, with one logical form per example.
    * Useful for reusing existing methods for validating types and action spaces, etc.
    */
  def getCcgDataset(dataset: Seq[WikiTablesExample]): Seq[CcgExample] = {
    val ccgDataset = for {
      example <- dataset
      lf <- example.logicalForms
      ccgExample = new CcgExample(example.sentence, null, null, lf)
    } yield {
      ccgExample
    }
    ccgDataset
  }

  def loadDataset(
    filename: String,
    derivationsPath: String,
    derivationsLimit: Int
  ): Seq[WikiTablesExample] = {
    val preprocessedFile = filename + preprocessingSuffix
    val dataset = if (Files.exists(Paths.get(preprocessedFile))) {
      readDatasetFromJson(preprocessedFile)
    } else {
      val sempreDataset = WikiTablesDataProcessor.getDataset(
        filename,
        true,
        true,
        derivationsPath,
        100,
        MAX_DERIVATIONS
      ).asScala
      val pnpDataset = sempreDataset.map(convertCustomExampleToWikiTablesExample)
      saveDatasetToJson(pnpDataset, preprocessedFile)
      pnpDataset
    }

    // Limit number of derivations if need be
    if (derivationsLimit >= 0 && derivationsLimit < MAX_DERIVATIONS) {
      for {
        ex <- dataset
        lfs = ex.possibleLogicalForms
      } yield {
        if (lfs.size > derivationsLimit) {
          ex.copy(possibleLogicalForms = lfs.toSeq.sortBy(_.size).take(derivationsLimit).toSet)
        } else {
          ex
        }
      }
    } else {
      dataset
    }
  }
  
  def loadDatasets(
      filenames: Seq[String],
      derivationsPath: String,
      derivationsLimit: Int
    ): Vector[RawExample] = {
    // The entity linker can become a parameter in the future
    // if it starts accepting parameters.
    val entityLinker = new WikiTablesEntityLinker()
    val trainingData = filenames.flatMap { filename => 
      val examples = loadDataset(filename, derivationsPath, derivationsLimit)
      val linkings = entityLinker.loadDataset(filename, examples)
      val tables = Table.loadDataset(filename, examples)
      
      val linkingsMap = linkings.map(x => (x.id, x)).toMap
      val tablesMap = tables.map(x => (x.id, x)).toMap
      examples.map(x => RawExample(x, linkingsMap(x.id), tablesMap(x.tableString)))
    }

    trainingData.toVector
  }

  def preprocessExample(
    example: RawExample,
    vocab: IndexedList[String],
    featureGenerator: SemanticParserFeatureGenerator,
    typeDeclaration: WikiTablesTypeDeclaration
  ) {
    // All we do here is add some annotations to the example.  Those annotations are:
    // 1. Token ids, computed using the vocab
    // 2. An EntityLinking
    
    // Each example has its own vocabulary which is comprised of the
    // parser's vocab plus new entries for OOV tokens found in
    // this example's question or entities.
    val exampleVocab = IndexedList.create[String]
    def tokenToId(token: String): Int = {
      if (vocab.contains(token)) {
        vocab.getIndex(token)
      } else {
        exampleVocab.add(token)
        vocab.size() + exampleVocab.getIndex(token)
      }
    }

    // Use UNK in the tokens to identify which tokens were OOV.
    // However, each UNKed token is assigned a distinct token id.
    val words = example.ex.sentence.getWords().asScala
    val unkedWords = words.map(x => if (vocab.contains(x)) x else UNK)
    val tokenIds = words.map(tokenToId(_)).toArray

    // Compute an entity linking.
    val entityLinking = example.linking.toEntityLinking(example.ex, tokenToId,
        featureGenerator, example.table, typeDeclaration)

    val annotations = example.ex.sentence.getAnnotations()
    annotations.put("originalTokens", example.ex.sentence.getWords().asScala.toList)
    annotations.put("unkedTokens", unkedWords.toList)
    annotations.put("tokenIds", tokenIds)
    annotations.put("entityLinking", entityLinking)
  }

  def toPnpLogicalForm(expression: Formula): Expression2 = {
    /*
    Sempre's lambda expressions are written differently from what pnp expects. We make the following changes
    1. Sempre uses ! and reverse interchangeably. Converting all ! to reverse.
      Eg.: (!fb:row.row.score (...)) -> ((reverse fb:row.row.score) (...))
    2. Variables in lambda forms are written without parentheses as arguments, and when they are actually
      used, declared as functions with 'var'.
      Eg.: (lambda x ((reverse fb:cell.cell.number) (var x))) -> (lambda (x) ((reverse fb:cell.cell.number) x))
      We need to do this for all bound variables.
     */
    var expressionString = expression.toString()
    // Change 1:
    expressionString = expressionString.replaceAll("!(fb:[^ ]*)", "(reverse $1)")
    val expressionTree = expression.toLispTree()

    val boundVariables = new mutable.HashSet[String]()
    // BFS to find all the free variables
    val fringe = new LinkedList[LispTree]()
    fringe.add(expressionTree)
    while (!fringe.isEmpty) {
      val fringeHead = fringe.remove()
      val fringeHeadFormula = Formulas.fromLispTree(fringeHead)
      if (fringeHeadFormula.isInstanceOf[LambdaFormula]) {
        boundVariables.add((fringeHeadFormula.asInstanceOf[LambdaFormula]).`var`)
      }
      if (!fringeHead.isLeaf()) {
        for (subTree <- fringeHead.children.asScala) {
          fringe.add(subTree)
        }
      }
    }

    for (variable <- boundVariables) {
      expressionString = expressionString.replaceAll(String.format("lambda %s", variable),
                                                     String.format("lambda (%s)", variable))
      expressionString = expressionString.replaceAll(String.format("\\(var %s\\)", variable), variable)
    }

    return ExpressionParser.expression2().parse(expressionString)
  }

  def toSempreLogicalForm(expression: Expression2): String = {
    val simplifier = new ExpressionSimplifier(Lists.newArrayList(new VariableCanonicalizationReplacementRule()))
    val simplifiedExpression = simplifier.apply(expression)

    val variableNames = new LinkedList[String](Seq("x", "y", "z").asJava)
    for (variable <- 'a' to 'w') {
      variableNames.add(variable.toString)
    }
    // Find all canonicalized bound variables
    val variableMap = new mutable.HashMap[String, String]()
    val fringe = new LinkedList[Expression2]()
    fringe.add(simplifiedExpression)
    while (!fringe.isEmpty()) {
      val currentExpression = fringe.remove()
      val currentChildren = currentExpression.getSubexpressions()
      if (currentChildren != null) {
        for (subExpression <- currentChildren.asScala)
          fringe.add(subExpression)
      }
      if (StaticAnalysis.isLambda(currentExpression)) {
        for (variable <- StaticAnalysis.getLambdaArguments(currentExpression).asScala) {
          variableMap.put(variable, variableNames.remove())
        }
      }
    }

    var expressionString = simplifiedExpression.toString()
    for (variable <- variableMap.keySet) {
      val variableName = variableMap(variable)
      expressionString = expressionString.replaceAll(String.format("lambda \\(%s\\)", Pattern.quote(variable)),
                                                     String.format("lambda %s", variableName))
      expressionString = expressionString.replaceAll(String.format("%s", Pattern.quote(variable)),
                                                     String.format("(var %s)", variableName))

      /*
      // XXX: test this
      // The last replacement can potentially lead to formulae like ((reverse fb:row.row.player) ((var x))))
      // with a single child in subtree with (var x). Fixing those.
      expressionString = expressionString.replaceAll(String.format("\\(\\(var %s\\)\\)", variableName),
                                                     String.format("(var %s)", variableName))
       */
    }
    expressionString
  }
}
