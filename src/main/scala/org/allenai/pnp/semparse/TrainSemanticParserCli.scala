package org.allenai.pnp.semparse

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

import org.allenai.pnp.Env
import org.allenai.pnp.LoglikelihoodTrainer
import org.allenai.pnp.GlobalLoglikelihoodTrainer
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
import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._
import joptsimple.OptionParser
import joptsimple.OptionSet
import joptsimple.OptionSpec
import org.allenai.pnp.BsoTrainer

/** Command line program for training a semantic parser.
  */
class TrainSemanticParserCli extends AbstractCli() {

  import TrainSemanticParserCli._
  
  var trainingDataOpt: OptionSpec[String] = null
  var entityDataOpt: OptionSpec[String] = null
  var testDataOpt: OptionSpec[String] = null
  var modelOutOpt: OptionSpec[String] = null
  
  override def initializeOptions(parser: OptionParser): Unit = {
    trainingDataOpt = parser.accepts("trainingData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',').required()
    entityDataOpt = parser.accepts("entityData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',').required()
    testDataOpt = parser.accepts("testData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',')
    modelOutOpt = parser.accepts("modelOut").withRequiredArg().ofType(classOf[String]).required()
  }
  
  override def run(options: OptionSet): Unit = {
    initialize(SemanticParserUtils.DYNET_PARAMS)
    
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

    println(trainingData.size + " training examples")
    println(entityData.size + " entity names")
    val wordCounts = SemanticParserUtils.getWordCounts(trainingData)
    val entityWordCounts = SemanticParserUtils.getWordCounts(entityData)
    
    // Vocab consists of all words that appear more than once in
    // the training data or appear in the entity names.
    val vocab = IndexedList.create(wordCounts.getKeysAboveCountThreshold(1.9))
    vocab.addAll(entityWordCounts.getKeysAboveCountThreshold(0.0))
    vocab.add(UNK)
    vocab.add(ENTITY)
    
    val entityDict = buildEntityDictionary(entityData, vocab, typeDeclaration)
    
    val trainPreprocessed = trainingData.map(x => preprocessExample(x, simplifier, vocab, entityDict)) 

    val actionSpace = ActionSpace.fromExpressions(
        trainPreprocessed.map(_.getLogicalForm), typeDeclaration, true)

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
    
    val model = PpModel.init(true)
    val parser = SemanticParser.create(actionSpace, vocab, model)
    
    println("*** Validating types ***")
    SemanticParserUtils.validateTypes(trainPreprocessed, typeDeclaration)
    println("*** Validating train set action space ***")
    SemanticParserUtils.validateActionSpace(trainPreprocessed, parser, typeDeclaration)
    println("*** Training ***")
    train(trainPreprocessed, parser, typeDeclaration)
    
    // Serialize model to disk.
    val saver = new ModelSaver(options.valueOf(modelOutOpt))
    model.save(saver)
    parser.save(saver)
    saver.done()
  }

  /** Train the parser by maximizing the likelihood of examples.
    * The model inside {@code parser} is used as the initial
    * parameters and is updated by this method to contain the
    * trained parameters.  
    */
  def train(examples: Seq[CcgExample], parser: SemanticParser,
      typeDeclaration: TypeDeclaration): Unit = {
    
    parser.dropoutProb = 0.5
    val ppExamples = for {
      x <- examples
      sent = x.getSentence
      tokenIds = sent.getAnnotation("tokenIds").asInstanceOf[Array[Int]]
      entityLinking = sent.getAnnotation("entityLinking").asInstanceOf[EntityLinking]
      unconditional = parser.generateExpression(tokenIds, entityLinking)
      oracle <- parser.generateExecutionOracle(x.getLogicalForm, entityLinking, typeDeclaration)
    } yield {
      PpExample(unconditional, unconditional, Env.init, oracle)
    }

    // Train model
    val model = parser.model
    val sgd = new SimpleSGDTrainer(model.model, 0.1f, 0.01f)
    val trainer = new LoglikelihoodTrainer(50, 100, false, model, sgd, new DefaultLogFunction())
    println("Running locally-normalized training...")
    trainer.train(ppExamples.toList)
    
    // Globally normalized training
    /*
    model.locallyNormalized = false
    val sgd2 = new SimpleSGDTrainer(model.model, 0.1f, 0.01f)
    val gtrainer = new BsoTrainer(50, 11, 50, model, sgd2, new DefaultLogFunction())
    println("Running globally-normalized training...")
    gtrainer.train(ppExamples.toList)
    */

    parser.dropoutProb = -1
  }
}

object TrainSemanticParserCli {
  
  val UNK = "<UNK>"
  val ENTITY = "<ENTITY>"

  def main(args: Array[String]): Unit = {
    (new TrainSemanticParserCli()).run(args)
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
        x => if (vocab.contains(x)) { x } else { UNK })
    val tokenIds = unkedWords.map(x => vocab.getIndex(x)).toList
    val entityLinking = entityDict.link(tokenIds)
    
    val entityAnonymizedWords = unkedWords.toArray
    val entityAnonymizedTokenIds = tokenIds.toArray
    for (entityMatch <- entityLinking.matches) {
      val span = entityMatch._1
      for (i <- span.start until span.end) {
        entityAnonymizedTokenIds(i) = vocab.getIndex(ENTITY)
        entityAnonymizedWords(i) = ENTITY
      }
    }

    val annotations = Maps.newHashMap[String, Object](sent.getAnnotations)
    annotations.put("originalTokens", sent.getWords.asScala.toList)
    annotations.put("tokenIds", entityAnonymizedTokenIds.toArray)
    annotations.put("entityLinking", entityLinking)

    val unkedSentence = new AnnotatedSentence(entityAnonymizedWords.toList.asJava,
        sent.getPosTags, annotations)
    
    new CcgExample(unkedSentence, ex.getDependencies, ex.getSyntacticParse, 
          simplifier.apply(ex.getLogicalForm))
  }
}
