package org.allenai.pnp.semparse

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.Map

import org.allenai.pnp.Env
import org.allenai.pnp.LoglikelihoodTrainer
import org.allenai.pnp.PpExample
import org.allenai.pnp.PpModel

import com.google.common.collect.Maps
import com.jayantkrish.jklol.ccg.CcgExample
import com.jayantkrish.jklol.ccg.cli.TrainSemanticParser
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.ccg.lambda2.ExpressionComparator
import com.jayantkrish.jklol.ccg.lambda2.ExpressionSimplifier
import com.jayantkrish.jklol.ccg.lambda2.SimplificationComparator
import com.jayantkrish.jklol.ccg.lambda2.StaticAnalysis
import com.jayantkrish.jklol.cli.AbstractCli
import com.jayantkrish.jklol.experiments.geoquery.GeoqueryUtil
import com.jayantkrish.jklol.nlpannotation.AnnotatedSentence
import com.jayantkrish.jklol.training.DefaultLogFunction
import com.jayantkrish.jklol.training.NullLogFunction
import com.jayantkrish.jklol.util.CountAccumulator
import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._
import joptsimple.OptionParser
import joptsimple.OptionSet
import joptsimple.OptionSpec
import com.jayantkrish.jklol.ccg.lambda.Type

/** Command line program for training a semantic parser.
  */
class SemanticParserCli extends AbstractCli() {
  
  var trainingDataOpt: OptionSpec[String] = null
  var entityDataOpt: OptionSpec[String] = null
  var testDataOpt: OptionSpec[String] = null
  
  override def initializeOptions(parser: OptionParser): Unit = {
    trainingDataOpt = parser.accepts("trainingData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',').required()
    entityDataOpt = parser.accepts("entityData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',').required()
    testDataOpt = parser.accepts("testData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',')
  }
  
  override def run(options: OptionSet): Unit = {
    myInitialize()
    
    // TODO: 
    // make templates have the most general type for arguments
    // allow generation of subtypes (e.g., a hole of type "lo" can be subtyped to a "c")
    // how do polymorphic types work in this context? -> generate if exists binding of variables such that it's a subtype?
    
    // Initialize expression processing for Geoquery logical forms. 
    val typeDeclaration = GeoqueryUtil.getSimpleTypeDeclaration()
    val simplifier = GeoqueryUtil.getExpressionSimplifier
    val comparator = new SimplificationComparator(simplifier)
    
    // Read and preprocess data
    val trainingData = ListBuffer[CcgExample]()
    for (filename <- options.valuesOf(trainingDataOpt).asScala) {
      trainingData ++= TrainSemanticParser.readCcgExamples(filename).asScala
    }
    
    val entityData = ListBuffer[CcgExample]()
    for (filename <- options.valuesOf(entityDataOpt).asScala) {
      entityData ++= TrainSemanticParser.readCcgExamples(filename).asScala
    }

    val testData = ListBuffer[CcgExample]()
    if (options.has(testDataOpt)) {
      for (filename <- options.valuesOf(testDataOpt).asScala) {
        testData ++= TrainSemanticParser.readCcgExamples(filename).asScala
      }
    }
    
    println(trainingData.size + " training examples")
    println(entityData.size + " entity names")
    println(testData.size + " test examples")
    val wordCounts = getWordCounts(trainingData)
    val entityWordCounts = getWordCounts(entityData)
    
    // Vocab consists of all words that appear more than once in
    // the training data or appear in the entity names.
    val vocab = IndexedList.create(wordCounts.getKeysAboveCountThreshold(1.9))
    vocab.addAll(entityWordCounts.getKeysAboveCountThreshold(0.0))
    vocab.add(SemanticParserCli.UNK)
    vocab.add(SemanticParserCli.ENTITY)
    
    val entityDict = buildEntityDictionary(entityData, vocab, typeDeclaration)
    
    val trainPreprocessed = trainingData.map(x => preprocessExample(x, simplifier, vocab, entityDict)) 
    val testPreprocessed = testData.map(x => preprocessExample(x, simplifier, vocab, entityDict))

    val actionSpace = SemanticParser.generateActionSpace(
        trainPreprocessed.map(_.getLogicalForm), typeDeclaration)
        
    // Remove entities from the action space, but ensure that there is
    // at least one valid action per type
    for (t <- actionSpace.allTypes) {
      actionSpace.typeTemplateMap.addBinding(t,
          ConstantTemplate(t, Expression2.constant("DUMMY:" + t)))
    }
    for (entity <- entityDict.map.values.flatten) {
      actionSpace.typeTemplateMap.removeBinding(entity.t, entity.template)
    }

    // println(actionSpace.rootTypes)
    // println(actionSpace.typeTemplateMap)
    
    val parser = new SemanticParser(actionSpace, vocab)
    
    println("*** Validating types ***")
    validateTypes(trainPreprocessed, typeDeclaration)
    println("*** Validating train set action space ***")
    validateActionSpace(trainPreprocessed, parser, typeDeclaration)
    println("*** Validating test set action space ***")
    validateActionSpace(testPreprocessed, parser, typeDeclaration)
    val trainedModel = train(trainPreprocessed, parser, typeDeclaration)
    println("***************** TEST EVALUATION *****************")
    val testResults = test(testPreprocessed, parser, trainedModel, typeDeclaration, simplifier, comparator)
    println("***************** TRAIN EVALUATION *****************")
    val trainResults = test(trainPreprocessed, parser, trainedModel, typeDeclaration, simplifier, comparator)
    
    println("")
    println("Test: ")
    println(testResults)
    println("Train: ")
    println(trainResults)
    
    // TODO: serialization
  }
  
  def getWordCounts(examples: Seq[CcgExample]): CountAccumulator[String] = {
    val acc = CountAccumulator.create[String]
    for (ex <- examples) {
      ex.getSentence.getWords.asScala.map(x => acc.increment(x, 1.0)) 
    }
    acc
  }
  
  def buildEntityDictionary(examples: Seq[CcgExample], vocab: IndexedList[String],
      typeDeclaration: TypeDeclaration): EntityDict = {
    val entityNames = ListBuffer[(Expression2, List[Int])]()
    for (ex <- examples) {
      val encodedName = ex.getSentence.getWords.asScala.toList.map(vocab.getIndex(_))
      entityNames += ((ex.getLogicalForm, encodedName))
    }

    val entityNameMap = SemanticParser.seqToMultimap(entityNames)
    val entityDict = ListBuffer[(List[Int], Entity)]()
    for (e <- entityNameMap.keySet) {
      val names = entityNameMap(e).toList
      val t = StaticAnalysis.inferType(e, typeDeclaration)
      val template = ConstantTemplate(t, e)
      val entity = Entity(e, t, template, names)
      for (name <- names) {
        entityDict += ((name, entity))
      }
    }

    new EntityDict(SemanticParser.seqToMultimap(entityDict))
  }

  def preprocessExample(ex: CcgExample, simplifier: ExpressionSimplifier,
      vocab: IndexedList[String], entityDict: EntityDict): CcgExample = {
    val sent = ex.getSentence
    val unkedWords = sent.getWords.asScala.map(
        x => if (vocab.contains(x)) { x } else { SemanticParserCli.UNK })
    val tokenIds = unkedWords.map(x => vocab.getIndex(x)).toList
    val entityLinking = entityDict.link(tokenIds)
    
    val entityAnonymizedWords = unkedWords.toArray
    val entityAnonymizedTokenIds = tokenIds.toArray
    for (entityMatch <- entityLinking.matches) {
      val span = entityMatch._1
      for (i <- span.start until span.end) {
        entityAnonymizedTokenIds(i) = vocab.getIndex(SemanticParserCli.ENTITY)
        entityAnonymizedWords(i) = SemanticParserCli.ENTITY
      }
    }

    val annotations = Maps.newHashMap[String, Object](sent.getAnnotations)
    annotations.put("originalTokens", sent.getWords.asScala.toList)
    annotations.put("tokenIds", entityAnonymizedTokenIds.toList)
    annotations.put("entityLinking", entityLinking)

    val unkedSentence = new AnnotatedSentence(entityAnonymizedWords.toList.asJava,
        sent.getPosTags, annotations)
    
    new CcgExample(unkedSentence, ex.getDependencies, ex.getSyntacticParse, 
          simplifier.apply(ex.getLogicalForm))
  }
  
  def validateTypes(examples: Seq[CcgExample], typeDeclaration: TypeDeclaration): Unit = {
    for (e <- examples) {
      val lf = e.getLogicalForm
      val typeMap = StaticAnalysis.inferTypeMap(lf, TypeDeclaration.TOP, typeDeclaration).asScala 
      
      for (i <- 0 until lf.size()) {
        if (typeMap.contains(i)) {
          val t = typeMap(i)
          if (isBadType(t)) {
            println(lf)
            println("  B " + i + " " + t + " " + lf.getSubexpression(i))
          }
        }
      }
    }
  }

  def isBadType(t: Type): Boolean = {
    if (t.isAtomic) {
      if (t.hasTypeVariables || t.equals(TypeDeclaration.TOP) || t.equals(TypeDeclaration.BOTTOM)) {
        true
      } else {
        false
      }
    } else {
      return isBadType(t.getArgumentType) || isBadType(t.getReturnType)
    }
  }

  /** Verify that the parser can generate the logical form
    * in each training example when the search is constrained
    * by the execution oracle.  
    */
  def validateActionSpace(examples: Seq[CcgExample], parser: SemanticParser,
      typeDeclaration: TypeDeclaration): Unit = {
    val model = parser.getModel

    println("")
    var maxParts = 0
    var numFailed = 0
    val usedRules = ListBuffer[(Type, Template)]()
    for (e <- examples) {
      val sent = e.getSentence
      val tokenIds = sent.getAnnotation("tokenIds").asInstanceOf[List[Int]]
      val entityLinking = sent.getAnnotation("entityLinking").asInstanceOf[EntityLinking]

      val oracleOpt = parser.generateExecutionOracle(e.getLogicalForm, entityLinking, typeDeclaration)
      val dist = parser.parse(tokenIds, entityLinking)

      if (oracleOpt.isDefined) {
        val oracle = oracleOpt.get
        val cg = new ComputationGraph
        val results = dist.beamSearch(1, 50, Env.init, oracle,
            model.getInitialComputationGraph(cg), new NullLogFunction())
        if (results.executions.size != 1) {
          println("ERROR: " + e + " " + results)
          println("  " + e.getSentence.getWords)
          println("  " + e.getLogicalForm)
          println("  " + e.getSentence.getAnnotation("entityLinking"))

          numFailed += 1
        } else {
          val numParts = results.executions(0).value.parts.size
          maxParts = Math.max(numParts, maxParts)
          if (results.executions.length > 1) {
            println("MULTIPLE: " + results.executions.length + " " + e)
            println("  " + e.getSentence.getWords)
            println("  " + e.getLogicalForm)
            println("  " + e.getSentence.getAnnotation("entityLinking"))
          } else {
            // println("OK   : " + numParts + " " + " "
          }
        }
        cg.delete
        
        // Accumulate the rules used in each example
        usedRules ++= oracle.holeTypes.zip(oracle.templates)

        // Print out the rules used to generate this logical form.
        /*
        println(e.getLogicalForm)
        for (t <- oracle.templates) {
          println("  " + t)
        }
        */
        
      } else {
        println("ORACLE: " + e)
        println("  " + e.getSentence.getWords)
        println("  " + e.getLogicalForm)
        println("  " + e.getSentence.getAnnotation("entityLinking"))

        numFailed += 1
      }
    }
    println("max parts: " + maxParts)
    println("decoding failures: " + numFailed)
    
    val holeTypes = usedRules.map(_._1).toSet
    val countMap = Map[Type, CountAccumulator[Template]]()
    for (t <- holeTypes) {
      countMap(t) = CountAccumulator.create()
    }
    
    for ((t, template) <- usedRules) {
      countMap(t).increment(template, 1.0)
    }
    
    for (t <- holeTypes) {
      println(t)
      val counts = countMap(t)
      for (template <- counts.getSortedKeys.asScala) {
        val count = counts.getCount(template)
        println("  " + count + " " + template) 
      }
    }
  }
  
  /** Train the parser by maximizing the likelihood of examples.
    * Returns a model with the trained parameters. 
    */
  def train(examples: Seq[CcgExample], parser: SemanticParser,
      typeDeclaration: TypeDeclaration): PpModel = {
    
    parser.dropoutProb = 0.5
    val ppExamples = for {
      x <- examples
      sent = x.getSentence
      tokenIds = sent.getAnnotation("tokenIds").asInstanceOf[List[Int]]
      entityLinking = sent.getAnnotation("entityLinking").asInstanceOf[EntityLinking]
      unconditional = parser.generateExpression(tokenIds, entityLinking)
      oracle <- parser.generateExecutionOracle(x.getLogicalForm, entityLinking, typeDeclaration)
    } yield {
      PpExample(unconditional, unconditional, Env.init, oracle)
    }

    // Train model
    val model = parser.getModel
    val sgd = new SimpleSGDTrainer(model.model, 0.1f, 0.01f)
    val trainer = new LoglikelihoodTrainer(50, 100, model, sgd, new DefaultLogFunction())
    trainer.train(ppExamples.toList)

    parser.dropoutProb = -1
    model
  }

  /** Evaluate the test accuracy of parser on examples. Logical
    * forms are compared for equality using comparator.  
    */
  def test(examples: Seq[CcgExample], parser: SemanticParser,
      model: PpModel, typeDeclaration: TypeDeclaration, simplifier: ExpressionSimplifier,
      comparator: ExpressionComparator): SemanticParserLoss = {
    println("")
    var numCorrect = 0
    var numCorrectAt10 = 0
    for (e <- examples) {
      println(e.getSentence.getWords.asScala.mkString(" "))
      println(e.getSentence.getAnnotation("originalTokens").asInstanceOf[List[String]].mkString(" "))
      println("expected: " + e.getLogicalForm)
      
      val sent = e.getSentence
      val dist = parser.parse(
          sent.getAnnotation("tokenIds").asInstanceOf[List[Int]],
          sent.getAnnotation("entityLinking").asInstanceOf[EntityLinking])
      val cg = new ComputationGraph
      val results = dist.beamSearch(10, 75, Env.init, null,
          model.getInitialComputationGraph(cg), new NullLogFunction())
          
      val beam = results.executions.slice(0, 10)
      val correct = beam.map { x =>
        val simplified = simplifier.apply(x.value.decodeExpression)
        if (comparator.equals(e.getLogicalForm, simplified)) {
          println("* " + x.logProb.formatted("%02.3f") + "  " + simplified)
          true
        } else {
          println("  " + x.logProb.formatted("%02.3f") + "  " + simplified)
          false
        }
      }
      
      if (correct.length > 0 && correct(0)) {
        numCorrect += 1
      }
      if (correct.fold(false)(_ || _)) {
        numCorrectAt10 += 1
      }
      
      // Print the attentions of the best predicted derivation
      val state = beam(0).value
      val templates = state.getTemplates
      val attentions = state.getAttentions
      val tokens = e.getSentence.getWords.asScala.toArray
      for (i <- 0 until templates.length) {
        val floatVector = as_vector(cg.get_value(attentions(i)))
        val values = for {
          j <- 0 until floatVector.size().asInstanceOf[Int]
        } yield {
          floatVector.get(j)
        }
        
        val maxIndex = values.zipWithIndex.max._2
        
        val tokenStrings = for {
          j <- 0 until values.length
        } yield {
          val color = if (j == maxIndex) {
            Console.RED
          } else if (values(j) > 0.1) {
            Console.YELLOW
          } else {
            Console.RESET
          }
          
          color + tokens(j) + Console.RESET
        }

        println("  " + tokenStrings.mkString(" ") + " " + templates(i))
      }
      
      cg.delete
    }
    
    val loss = SemanticParserLoss(numCorrect, numCorrectAt10, examples.length)
    println(loss)
    loss
  }
}

object SemanticParserCli {
  
  val UNK = "<UNK>"
  val ENTITY = "<ENTITY>"

  def main(args: Array[String]): Unit = {
    (new SemanticParserCli()).run(args)
  }
}

case class SemanticParserLoss(numCorrect: Int, oracleNumCorrect: Int, numExamples: Int) {
  val accuracy: Double = numCorrect.asInstanceOf[Double] / numExamples
  val oracleAccuracy: Double = oracleNumCorrect.asInstanceOf[Double] / numExamples
  
  override def toString(): String = {
    "accuracy: " + accuracy + " " + numCorrect + " / " + numExamples + "\n" +
    "oracle  : " + oracleAccuracy + " " + oracleNumCorrect + " / " + numExamples  
  }
}