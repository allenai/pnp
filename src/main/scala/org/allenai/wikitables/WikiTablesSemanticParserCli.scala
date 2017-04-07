package org.allenai.wikitables

import scala.collection.JavaConverters._
import scala.io.Source

import org.allenai.pnp.BsoTrainer
import org.allenai.pnp.Env
import org.allenai.pnp.LoglikelihoodTrainer
import org.allenai.pnp.LoglikelihoodTrainer
import org.allenai.pnp.PnpExample
import org.allenai.pnp.PnpModel
import org.allenai.pnp.semparse.ActionSpace
import org.allenai.pnp.semparse.ApplicationTemplate
import org.allenai.pnp.semparse.ConstantTemplate
import org.allenai.pnp.semparse.EntityLinking
import org.allenai.pnp.semparse.SemanticParser
import org.allenai.pnp.semparse.SemanticParserConfig
import org.allenai.pnp.semparse.SemanticParserUtils
import org.allenai.pnp.semparse.Template

import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import com.jayantkrish.jklol.ccg.lambda.Type
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.ccg.lambda2.ExpressionSimplifier
import com.jayantkrish.jklol.ccg.lambda2.SimplificationComparator
import com.jayantkrish.jklol.ccg.lambda2.StaticAnalysis
import com.jayantkrish.jklol.cli.AbstractCli
import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.dynet._
import joptsimple.OptionParser
import joptsimple.OptionSet
import joptsimple.OptionSpec
import com.jayantkrish.jklol.util.IndexedList
import scala.util.Random
import scala.collection.mutable.ListBuffer
import org.allenai.pnp.LoglikelihoodTrainer
import org.allenai.pnp.BsoTrainer

/** Command line program for training a semantic parser
  * on the WikiTables data set.
  * runMain org.allenai.wikitables.WikiTablesSemanticParserCli -trainingData TRAIN-DATA-PATH
  *                                                           [-testData TEST-DATA-PATH]
  *                                                           [-derivationsPath PATH-TO-LOGICAL-FORMS]
  * If derivationsPath is not specified, Sempre will be used to parse utterances (this will be SLOW!)
  */
class WikiTablesSemanticParserCli extends AbstractCli() {

  import WikiTablesUtil._

  var trainingDataOpt: OptionSpec[String] = null
  var devDataOpt: OptionSpec[String] = null
  // Path to the directory containing the correct logical forms
  var derivationsPathOpt: OptionSpec[String] = null
  // Path for the final model. 
  var modelOutputOpt: OptionSpec[String] = null
  // Directory for intermediate models produced during training.
  var modelOutputDirOpt: OptionSpec[String] = null
  // Word embeddings to initialize the model's parameters.
  var wordEmbeddingsOpt: OptionSpec[String] = null
  
  // Semantic parser configuration
  var inputDimOpt: OptionSpec[Integer] = null
  var hiddenDimOpt: OptionSpec[Integer] = null
  var actionDimOpt: OptionSpec[Integer] = null
  var actionHiddenDimOpt: OptionSpec[Integer] = null

  var maxDerivationsOpt: OptionSpec[Integer] = null
  var vocabThreshold: OptionSpec[Integer] = null  
  var epochsOpt: OptionSpec[Integer] = null
  var dropoutOpt: OptionSpec[Double] = null
  var beamSizeOpt: OptionSpec[Integer] = null
  var devBeamSizeOpt: OptionSpec[Integer] = null
  var lasoOpt: OptionSpec[Void] = null
  var editDistanceOpt: OptionSpec[Void] = null
  var actionBiasOpt: OptionSpec[Void] = null
  var reluOpt: OptionSpec[Void] = null
  var actionLstmHiddenLayerOpt: OptionSpec[Void] = null
  var concatLstmForDecoderOpt: OptionSpec[Void] = null
  var maxPoolEntityTokenSimilaritiesOpt: OptionSpec[Void] = null

  var skipActionSpaceValidationOpt: OptionSpec[Void] = null
  var trainOnAnnotatedLfsOpt: OptionSpec[Void] = null
  var seq2TreeOpt: OptionSpec[Void] = null
  var seq2SeqOpt: OptionSpec[Void] = null

  // Initialize expression processing for Wikitables logical forms.
  val simplifier = ExpressionSimplifier.lambdaCalculus()
  val comparator = new SimplificationComparator(simplifier)
  val logicalFormParser = ExpressionParser.expression2()

  override def initializeOptions(parser: OptionParser): Unit = {
    trainingDataOpt = parser.accepts("trainingData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',').required()
    devDataOpt = parser.accepts("devData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',')
    derivationsPathOpt = parser.accepts("derivationsPath").withRequiredArg().ofType(classOf[String])
    modelOutputOpt = parser.accepts("modelOut").withRequiredArg().ofType(classOf[String]).required()
    modelOutputDirOpt = parser.accepts("modelDir").withRequiredArg().ofType(classOf[String])
    wordEmbeddingsOpt = parser.accepts("wordEmbeddings").withRequiredArg().ofType(classOf[String])
    
    inputDimOpt = parser.accepts("inputDim").withRequiredArg().ofType(classOf[Integer]).defaultsTo(200)
    hiddenDimOpt = parser.accepts("hiddenDim").withRequiredArg().ofType(classOf[Integer]).defaultsTo(100)
    actionDimOpt = parser.accepts("actionDim").withRequiredArg().ofType(classOf[Integer]).defaultsTo(100)
    actionHiddenDimOpt = parser.accepts("actionHiddenDim").withRequiredArg().ofType(classOf[Integer]).defaultsTo(100)
    
    maxDerivationsOpt = parser.accepts("maxDerivations").withRequiredArg().ofType(classOf[Integer]).defaultsTo(-1)
    // A word must appear *strictly more* times than this threshold to be included
    // in the vocabulary.
    vocabThreshold = parser.accepts("vocabThreshold").withRequiredArg().ofType(classOf[Integer]).defaultsTo(1)
    epochsOpt = parser.accepts("epochs").withRequiredArg().ofType(classOf[Integer]).defaultsTo(50)
    beamSizeOpt = parser.accepts("beamSize").withRequiredArg().ofType(classOf[Integer]).defaultsTo(5)
    devBeamSizeOpt = parser.accepts("devBeamSize").withRequiredArg().ofType(classOf[Integer]).defaultsTo(5)
    dropoutOpt = parser.accepts("dropout").withRequiredArg().ofType(classOf[Double]).defaultsTo(0.5)
    lasoOpt = parser.accepts("laso")
    editDistanceOpt = parser.accepts("editDistance")
    actionBiasOpt = parser.accepts("actionBias")
    reluOpt = parser.accepts("relu")
    actionLstmHiddenLayerOpt = parser.accepts("actionLstmHiddenLayer")
    concatLstmForDecoderOpt = parser.accepts("concatLstmForDecoder")
    maxPoolEntityTokenSimilaritiesOpt = parser.accepts("maxPoolEntityTokenSimilarities")

    skipActionSpaceValidationOpt = parser.accepts("skipActionSpaceValidation")
    trainOnAnnotatedLfsOpt = parser.accepts("trainOnAnnotatedLfs")
    seq2TreeOpt = parser.accepts("seq2Tree")
    seq2SeqOpt = parser.accepts("seq2Seq")
  }

  def initializeTrainingData(options: OptionSet, typeDeclaration: TypeDeclaration,
      preprocessor: LfPreprocessor,
      wordEmbeddings: Option[Array[(String, FloatVector)]]) = {
    // Read and preprocess data
    val trainingData = loadDatasets(options.valuesOf(trainingDataOpt).asScala,
        options.valueOf(derivationsPathOpt),
        options.valueOf(maxDerivationsOpt),
        preprocessor)

    println("Read " + trainingData.size + " training examples")
    val (vocab, vocabCounts) = computeVocabulary(trainingData, options.valueOf(vocabThreshold))
    val parserVocab = if (wordEmbeddings.isDefined) {
      IndexedList.create(wordEmbeddings.get.map(_._1).toVector.asJava)
    } else {
      vocab
    }

    val featureGenerator = SemanticParserFeatureGenerator.getWikitablesGenerator(
        options.has(editDistanceOpt), vocab, vocabCounts)

    // Eliminate those examples that Sempre did not find correct logical forms for,
    // and preprocess the remaining
    val filteredTrainingData = trainingData.filter(!_.ex.logicalForms.isEmpty)
        
    // preprocessExample modifies the `annotations` data structure in example.sentence, adding
    // some things to it.  We don't need a `map`, just a `foreach`.
    filteredTrainingData.foreach(x => preprocessExample(x, parserVocab, featureGenerator,
        typeDeclaration))
    println("Found correct logical forms for " + filteredTrainingData.size + " training examples")

    println("Preprocessed:")
    for (example <- filteredTrainingData) {
      println(example.ex.sentence.getWords)
      println(example.ex.logicalForms)
    }
    (filteredTrainingData.map(_.ex), parserVocab, featureGenerator)
  }

  def initializeDevelopmentData(options: OptionSet, typeDeclaration: TypeDeclaration,
      preprocessor: LfPreprocessor, featureGen: SemanticParserFeatureGenerator,
      vocab: IndexedList[String]): Seq[WikiTablesExample] = {
    val devData = loadDatasets(options.valuesOf(devDataOpt).asScala,
        options.valueOf(derivationsPathOpt), -1, preprocessor)

    println("Read " + devData.size + " development examples")

    devData.foreach(x => WikiTablesUtil.preprocessExample(x, vocab, featureGen,
        typeDeclaration))
    devData.map(_.ex)
  }

  override def run(options: OptionSet): Unit = {
    Initialize.initialize(Map("dynet-mem" -> "3096"))
 
    val typeDeclaration = if (options.has(seq2TreeOpt)) {
      new Seq2TreeTypeDeclaration()
    } else if (options.has(seq2SeqOpt)) {
      new Seq2SeqTypeDeclaration()
    } else {
      new WikiTablesTypeDeclaration()
    }

    val lfPreprocessor = if (options.has(seq2TreeOpt)) {
      new Seq2TreePreprocessor()
    } else if (options.has(seq2SeqOpt)) {
      new Seq2SeqPreprocessor()
    } else {
      new NullLfPreprocessor()
    }
    
    val wordEmbeddings = if (options.has(wordEmbeddingsOpt)) {
      Some(readWordEmbeddings(options.valueOf(wordEmbeddingsOpt)))
    } else {
      None
    }

    // Read training data
    val (trainingData, vocab, featureGenerator) = initializeTrainingData(options, typeDeclaration,
        lfPreprocessor, wordEmbeddings)

    // Read development data (if provided)
    val devData = initializeDevelopmentData(options, typeDeclaration,
        lfPreprocessor, featureGenerator, vocab)

    // Generate the action space of the semantic parser from the logical
    // forms that are well-typed.
    val trainingLfs = trainingData.map(_.logicalForms).flatten
    println("*** Validating types ***")
    val wellTypedTrainingLfs = trainingLfs.filter(lf =>
      SemanticParserUtils.validateTypes(lf, typeDeclaration))

    // Combine applications makes templates like <t> -> (f <x> <y>)
    // where f is a constant and <.> is a type. This is necessary 
    // to replicate a seq2tree / seq2seq model given the lf preprocessing
    val combineApplications = options.has(seq2TreeOpt) || options.has(seq2SeqOpt)
    val actionSpace = getActionSpace(wellTypedTrainingLfs, typeDeclaration, combineApplications)

    val model = PnpModel.init(true)
    val config = new SemanticParserConfig()
    config.inputDim = options.valueOf(inputDimOpt)
    config.hiddenDim = options.valueOf(hiddenDimOpt)
    config.actionDim = options.valueOf(actionDimOpt)
    config.actionHiddenDim = options.valueOf(actionHiddenDimOpt)
    config.featureGenerator = Some(featureGenerator)
    config.entityLinkingLearnedSimilarity = true
    // TODO: turn back on (?)
    config.encodeWithSoftEntityLinking = false
    config.distinctUnkVectors = true
    config.actionBias = options.has(actionBiasOpt)
    config.relu = options.has(reluOpt)
    config.actionLstmHiddenLayer = options.has(actionLstmHiddenLayerOpt)
    config.concatLstmForDecoder = options.has(concatLstmForDecoderOpt)
    config.maxPoolEntityTokenSimiliarities = options.has(maxPoolEntityTokenSimilaritiesOpt)
    val parser = SemanticParser.create(actionSpace, vocab, wordEmbeddings, config, model)

    if (!options.has(skipActionSpaceValidationOpt)) {
      val trainSeparatedLfs = getCcgDataset(trainingData)

      println("*** Validating train set action space ***")
      SemanticParserUtils.validateActionSpace(trainSeparatedLfs, parser, typeDeclaration)
    } else {
      println("Skipping action space validation")
    }

    val modelOutputDir = if (options.has(modelOutputDirOpt)) { 
      Some(options.valueOf(modelOutputDirOpt))
    } else {
      None
    }

    train(trainingData, devData, parser, typeDeclaration, simplifier, lfPreprocessor,
        options.valueOf(epochsOpt), options.valueOf(beamSizeOpt), options.valueOf(devBeamSizeOpt),
        options.valueOf(dropoutOpt), options.has(lasoOpt), modelOutputDir)

    val saver = new ModelSaver(options.valueOf(modelOutputOpt))
    model.save(saver)
    parser.save(saver)
    saver.done()
  }

  /** Train the parser by maximizing the likelihood of examples.
    * Returns a model with the trained parameters.
    */
  def train(trainingExamples: Seq[WikiTablesExample], devExamples: Seq[WikiTablesExample],
      parser: SemanticParser, typeDeclaration: TypeDeclaration,
      simplifier: ExpressionSimplifier, preprocessor: LfPreprocessor,
      epochs: Int, beamSize: Int, devBeamSize: Int,
      dropout: Double, laso: Boolean, modelDir: Option[String]): Unit = {

    parser.dropoutProb = dropout
    val pnpExamples = for {
      x <- trainingExamples
      sentence = x.sentence
      tokenIds = sentence.getAnnotation("tokenIds").asInstanceOf[Array[Int]]
      entityLinking = sentence.getAnnotation("entityLinking").asInstanceOf[EntityLinking]
      unconditional = parser.generateExpression(tokenIds, entityLinking)
      oracle <- if (laso) {
        parser.getMultiMarginScore(x.logicalForms, entityLinking, typeDeclaration)
      } else {
        parser.getMultiLabelScore(x.logicalForms, entityLinking, typeDeclaration)
      }
    } yield {
      PnpExample(unconditional, unconditional, Env.init, oracle)
    }

    println(pnpExamples.size + " training examples after oracle generation.")

    // If we have dev examples, subsample the same number of training examples
    // for evaluating parser accuracy as training progresses.
    // XXX: disabled because the preprocessing here can be very slow.
    /*
    val trainErrorExamples = if (devExamples.size > 0) {
      Random.shuffle(trainingExamples).slice(0, Math.min(devExamples.size, trainingExamples.size))
    } else {
      List()
    }
    */
    val trainErrorExamples: List[WikiTablesExample] = List()

    // Call .getContext on every example that we'll use during error
    // evaluation. This preprocesses the corresponding table using
    // corenlp and (I think) caches the result somewhere in Sempre.
    // This will happen during the error evaluation of training anyway,
    // but doing it up-front makes the training timers more useful. 
    println("Preprocessing context for train/dev evaluation examples.")
    // TODO: how much does the choice of training examples affect the train error results??
    // seems to affect stanford's processing time significantly.
    trainErrorExamples.foreach(x => x.getContext)
    devExamples.foreach(x => x.getContext)
    
    // Train model
    val model = parser.model
    val sgd = new SimpleSGDTrainer(model.model, 0.1f, 0.01f)
    val logFunction = new SemanticParserLogFunction(modelDir, parser, trainErrorExamples,
        devExamples, devBeamSize, 2, typeDeclaration, new SimplificationComparator(simplifier),
        preprocessor)
    
    if (laso) {
      println("Running LaSO training...")
      model.locallyNormalized = false
      val trainer = new BsoTrainer(epochs, beamSize, 50, model, sgd, logFunction)
      trainer.train(pnpExamples.toList)
    } else {
      println("Running loglikelihood training...")
      model.locallyNormalized = true
      val trainer = new LoglikelihoodTrainer(epochs, beamSize, true, model, sgd,
          logFunction)
      trainer.train(pnpExamples.toList)
    }
    parser.dropoutProb = -1
  }
  
  def getActionSpace(wellTypedTrainingLfs: Seq[Expression2], typeDeclaration: TypeDeclaration,
      combineApplications: Boolean): ActionSpace = {
    println("*** Generating action space ***")
    var actionSpace = ActionSpace.fromExpressions(wellTypedTrainingLfs, typeDeclaration, combineApplications)

    // Remove specific numbers/rows/cells from the action space.
    // These need to be added back in on a per-table basis.
    val filterTypes = Seq(Type.parseFrom("i"), Type.parseFrom("c"), Type.parseFrom("p"), Type.parseFrom("<c,r>"))
    for (t <- filterTypes) {
      val templates = actionSpace.typeTemplateMap.getOrElse(t, List()).toSet
      for (template <- templates) {
        if (template.isInstanceOf[ConstantTemplate]) {
          actionSpace.typeTemplateMap.removeBinding(t, template)
        }
      }

      // Create a dummy action for the type to ensure that at least
      // one valid action is always possible.
      if (actionSpace.typeTemplateMap.getOrElse(t, Set()).size == 0) {
        actionSpace.typeTemplateMap.addBinding(t,
            ConstantTemplate(t, Expression2.constant("DUMMY:" + t)))
      }
    }
    
    // Remove specific numbers/rows columns for the seq2tree and seq2seq
    // action spaces.
    val wikitablesTypes = new WikiTablesTypeDeclaration()
    val anyTemplates = actionSpace.typeTemplateMap.getOrElse(Seq2TreeTypeDeclaration.anyType, List()).toSet
    for (template <- anyTemplates) {
      if (template.isInstanceOf[ConstantTemplate]) {
        val c = template.asInstanceOf[ConstantTemplate]
        val t = StaticAnalysis.inferType(c.expr, wikitablesTypes)
        if (filterTypes.contains(t)) {
          actionSpace.typeTemplateMap.removeBinding(Seq2TreeTypeDeclaration.anyType, template)
        }
      }
    }
     
    val s2sTemplates = actionSpace.typeTemplateMap.getOrElse(Seq2SeqTypeDeclaration.endType, List()).toSet
    for (template <- s2sTemplates) {
      if (template.isInstanceOf[ApplicationTemplate]) {
        val a = template.asInstanceOf[ApplicationTemplate]
        val funcExpr = a.expr.getSubexpressions.get(0)
        val t = StaticAnalysis.inferType(funcExpr, wikitablesTypes)
        if (filterTypes.contains(t)) {
          actionSpace.typeTemplateMap.removeBinding(Seq2SeqTypeDeclaration.endType, template)
        }
      }
    }

    // Print out the action space
    for (t <- actionSpace.typeTemplateMap.keys) {
      println(t)
      for (template <- actionSpace.typeTemplateMap(t)) {
        println("  " + template)
      }
    }

    actionSpace
  }

  def readWordEmbeddings(filename: String): Array[(String, FloatVector)] = {
    val embeddingIter = Source.fromFile(filename).getLines().map { s => 
      val parts = s.split(" ")
      val token = parts(0)
      val floatVector = new FloatVector(parts.drop(1).map(x => x.toFloat))
      (token, floatVector)
    }

    embeddingIter.toArray
  }
}

object WikiTablesSemanticParserCli {

  def main(args: Array[String]): Unit = {
    (new WikiTablesSemanticParserCli()).run(args)
  }
}
