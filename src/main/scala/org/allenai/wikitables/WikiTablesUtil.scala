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
      ("NER" -> example.sentence.getAnnotation("NER").asInstanceOf[Seq[Seq[String]]]) ~
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
    val ner = (json \ "NER").extract[List[List[String]]]
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
    sentence.getAnnotations().put("NER", ner)
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
      if (tag == "O" && label == null) Seq() else Seq(tag, label)
    }}
    sentence.getAnnotations().put("NER", filteredNer)

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

  def getEntityTokenCounts(entityNames: List[String]): CountAccumulator[String] = {
    val acc = CountAccumulator.create[String]
    for (entityString <- entityNames) {
      tokenizeEntity(entityString).map(x => acc.increment(x, 1.0))
    }
    acc
  }

  def computeVocabulary(trainingDataWithEntities: Map[WikiTablesExample, Seq[Pair[Pair[Integer, Integer], Formula]]]) = {
    val wordCounts = getTokenCounts(trainingDataWithEntities.keys)
    val allEntities = trainingDataWithEntities.values.flatten.map(p => p.getSecond().toString).toList
    val entityCounts = getEntityTokenCounts(allEntities)
    // Vocab consists of all words that appear more than once in
    // the training data and in the name of any entity.
    val vocab = IndexedList.create(wordCounts.getKeysAboveCountThreshold(1.9))
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
    includeDerivationsForTrain: Boolean,
    derivationsPath: String,
    derivationsLimit: Int
  ): Seq[WikiTablesExample] = {
    val preprocessedFile = filename + preprocessingSuffix
    if (Files.exists(Paths.get(preprocessedFile))) {
      readDatasetFromJson(preprocessedFile)
    } else {
      val sempreDataset = WikiTablesDataProcessor.getDataset(
        filename,
        true,
        includeDerivationsForTrain,
        derivationsPath,
        100,
        derivationsLimit
      ).asScala
      val pnpDataset = sempreDataset.map(convertCustomExampleToWikiTablesExample)
      saveDatasetToJson(pnpDataset, preprocessedFile)
      pnpDataset
    }
  }

  /**
   * Converts the id of an entity to a sequence of
   * tokens in its "name." The tokens will have the form
   * [type, token1, token2, ..]. For example:
   * "fb:row.row.player_name" -> ["fb:row.row", "player", "name"]
   */
  def tokenizeEntity(entityString: String): List[String] = {
    val lastDotInd = entityString.lastIndexOf(".")
    val entityTokens = if (lastDotInd == -1) {
      entityString.split('_').toList
    } else {
      val (typeName, entityName) = entityString.splitAt(lastDotInd)
      val tokens = entityName.substring(1).split('_').toList
      // Return the type prefixed to the tokens
      // List(typeName) ++ tokens

      // Return only the tokens
      tokens
    }

    // println("tokens: " + entityString + " " + entityTokens)
    entityTokens
  }

  def sempreEntityLinkingToPnpEntityLinking(
    sempreEntityLinking: Seq[Pair[Pair[Integer, Integer], Formula]],
    tokenToId: String => Int,
    typeDeclaration: WikiTablesTypeDeclaration
    ): EntityLinking = {
    val builder = mutable.ListBuffer[(Option[Span], Entity, List[Int], Double)]()
    for (linking <- sempreEntityLinking) {
      val entityString = linking.getSecond.toString
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

        val span = if (linking.getFirst() != null) {
          val start = linking.getFirst().getFirst()
          val end = linking.getFirst().getSecond()
          Some(Span(start, end))
        } else {
          None
        }

        // Tokens in the names of entities are also encoded with the
        // example-specific vocabulary.
        val entityTokens = tokenizeEntity(entityString)
        val entityTokenIds = entityTokens.map(tokenToId(_)).toList
        val entity = Entity(entityExpr, entityType, template, List(entityTokenIds))
        builder += ((span, entity, entityTokenIds, 0.1))
      }
    }
    new EntityLinking(builder.toList)
  }

  def preprocessExample(
    example: WikiTablesExample,
    vocab: IndexedList[String],
    sempreEntityLinking: Seq[Pair[Pair[Integer, Integer], Formula]],
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
    val words = example.sentence.getWords().asScala
    val unkedWords = words.map(x => if (vocab.contains(x)) x else UNK)
    val tokenIds = words.map(tokenToId(_)).toArray

    // Compute an entity linking.
    val entityLinking = sempreEntityLinkingToPnpEntityLinking(sempreEntityLinking,
        tokenToId, typeDeclaration)

    val annotations = example.sentence.getAnnotations()
    annotations.put("originalTokens", example.sentence.getWords().asScala.toList)
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
