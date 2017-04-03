package org.allenai.pnp.semparse

import scala.collection.JavaConverters._
import org.allenai.pnp.{Env, Pnp, PnpInferenceContext, PnpModel}

import org.scalatest.FlatSpec
import org.scalatest.Matchers
import com.jayantkrish.jklol.ccg.lambda.ExplicitTypeDeclaration
import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import com.jayantkrish.jklol.training.NullLogFunction
import com.jayantkrish.jklol.util.IndexedList
import edu.cmu.dynet._
import com.jayantkrish.jklol.ccg.lambda2.StaticAnalysis
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.training.DefaultLogFunction
import org.allenai.pnp.LoglikelihoodTrainer
import org.allenai.pnp.PnpExample
import com.google.common.base.Preconditions
import com.jayantkrish.jklol.ccg.lambda2.ExpressionSimplifier
import com.jayantkrish.jklol.ccg.lambda2.SimplificationComparator
import org.allenai.pnp.BsoTrainer

class SemanticParserSpec extends FlatSpec with Matchers {
  
  Initialize.initialize()
 
  val dataStrings = List(
      ("state", List(), "state:<e,t>"),
      ("city", List(), "city:<e,t>"),
      ("biggest city", List(), "(argmax:<<e,t>,e> city:<e,t>)"),
      ("texas", List(), "texas:e"), 
      ("major city", List(), "(lambda ($0) (and:<t*,t> (city:<e,t> $0) (major:<e,t> $0)))"),
      ("city in texas", List(), "(lambda ($0) (and:<t*,t> (city:<e,t> $0) (loc:<e,<e,t>> $0 texas:e)))")
  )

  val exprParser = ExpressionParser.expression2()
  val typeDeclaration = ExplicitTypeDeclaration.getDefault()
  val simplifier = ExpressionSimplifier.lambdaCalculus()

  val vocab = IndexedList.create[String]
  for (d <- dataStrings) {
    vocab.addAll(d._1.split(" ").toList.asJava)
  }

  val data = dataStrings.map(x => encodeExample(x._1, x._2, x._3))

  val lexicon = ActionSpace.fromExpressions(data.map(_.lf), typeDeclaration, false)
  val model = PnpModel.init(true)
  val config = new SemanticParserConfig()
  val parser = SemanticParser.create(lexicon, vocab, config, model)
  
  /**
   * Creates a training example for the semantic parser given a
   * simplified string format that is easy to write down.
   */
  def encodeExample(tokenString: String, entities: List[(String, String)], lfString: String
      ): SemanticParserExample = {
    
    val myVocab = IndexedList.create[String]
    def tokenToId(token: String): Int = {
      if (vocab.contains(token)) {
        vocab.getIndex(token)
      } else {
        myVocab.add(token)
        vocab.size + myVocab.getIndex(token)
      }
    }
    
    val tokens = tokenString.split(" ")
    val tokenIds = tokens.map(tokenToId(_)).toArray
    
    val entityLinkingList = for {
      (entityName, entityLfString) <- entities
    } yield {
      val entityTokenIds = entityName.split(" ").map(tokenToId(_)).toList
      val entityLf = exprParser.parse(entityLfString)
      val entityType = StaticAnalysis.inferType(entityLf, typeDeclaration)
      val template = ConstantTemplate(entityType, entityLf)
      val entity = Entity(entityLf, entityType, template, List(entityTokenIds))
      
      val span: Option[Span] = None
     
      (span, entity, entityTokenIds, 0.1)
    }

    val lf = simplifier.apply(exprParser.parse(lfString)) 
    SemanticParserExample(tokenIds, EntityLinking(entityLinkingList), lf)  
  }
  
  /**
   * Train {@code parser} on a collection of examples.
   */
  def train(examples: Seq[SemanticParserExample], parser: SemanticParser): Unit = {
    val pnpExamples = for {
      ex <- examples
      unconditional = parser.generateExpression(ex.tokenIds, ex.entityLinking)
      oracle <- parser.getLabelScore(ex.lf, ex.entityLinking, typeDeclaration)
    } yield {
      PnpExample(unconditional, unconditional, Env.init, oracle)
    }
    Preconditions.checkState(pnpExamples.size == examples.size)

    // Train model
    val model = parser.model
    val sgd = new SimpleSGDTrainer(model.model, 0.1f, 0.01f)
    val trainer = new LoglikelihoodTrainer(50, 5, true, model, sgd,
        new NullLogFunction())
    trainer.train(pnpExamples.toList)
  }
  
  /**
   * Train {@code parser} on a collection of examples.
   */
  def trainLaso(examples: Seq[SemanticParserExample], parser: SemanticParser): Unit = {
    val pnpExamples = for {
      ex <- examples
      unconditional = parser.generateExpression(ex.tokenIds, ex.entityLinking)
      oracle <- parser.getMultiMarginScore(Seq(ex.lf), ex.entityLinking, typeDeclaration)
    } yield {
      PnpExample(unconditional, unconditional, Env.init, oracle)
    }
    Preconditions.checkState(pnpExamples.size == examples.size)

    // Train model
    val model = parser.model
    model.locallyNormalized = false
    val sgd = new SimpleSGDTrainer(model.model, 0.1f, 0.01f)
    val trainer = new BsoTrainer(50, 5, 20, model, sgd, new NullLogFunction())
    trainer.train(pnpExamples.toList)
  }
  
  /**
   * Assert that parser's highest-scoring predicted logical form
   * is correct for {@code example}.
   */
  def assertPredictionCorrect(example: SemanticParserExample, parser: SemanticParser): Unit = {
    ComputationGraph.renew()
    val context = PnpInferenceContext.init(parser.model)
    val results = parser.parse(example.tokenIds, example.entityLinking)
      .beamSearch(5, 50, Env.init, context).executions
          
    results.size should be > 0
    
    val predictedLf = simplifier.apply(results(0).value.decodeExpression)
    val correctLf = simplifier.apply(example.lf)    
    predictedLf should equal(correctLf)
  }

  "SemanticParser" should "decode expressions to template sequences" in {
    val e = exprParser.parse(
        "(argmax:<<e,t>,e> (lambda ($0) (and:<t*,t> (city:<e,t> $0) (major:<e,t> $0))))")
    // This method will throw an error if it can't decode the expression properly. 
    val templates = parser.generateActionSequence(e, EntityLinking(List()), typeDeclaration)
  }
  
  it should "condition on expressions" in {
    val label = exprParser.parse("(lambda ($0) (and:<t*,t> (city:<e,t> $0) (major:<e,t> $0)))")
    val entityLinking = EntityLinking(List())
    val oracle = parser.getLabelScore(label, entityLinking, typeDeclaration).get
    val exprs = parser.generateExpression(Array("major", "city").map(vocab.getIndex(_)),
        entityLinking)

    ComputationGraph.renew()
    val context = PnpInferenceContext.init(model).addExecutionScore(oracle)

    val results = exprs.beamSearch(1, -1, Env.init, context).executions
    results.length should be(1)
    results(0).value should equal(label)
  }
  
  it should "condition on multiple expressions" in {
    val label1 = exprParser.parse("(lambda ($0) (and:<t*,t> (city:<e,t> $0) (major:<e,t> $0)))")
    val label2 = exprParser.parse("(lambda ($0) (state:<e,t> $0))")
    val labels = Set(label1, label2)
    val entityLinking = EntityLinking(List())
    val oracle = parser.getMultiLabelScore(labels, entityLinking, typeDeclaration).get
    
    val exprs = parser.generateExpression(Array("major", "city").map(vocab.getIndex(_)),
        entityLinking)

    ComputationGraph.renew()
    val context = PnpInferenceContext.init(model).addExecutionScore(oracle)

    val results = exprs.beamSearch(2, -1, Env.init, context).executions
    results.length should be(2)
    results.map(_.value).toSet should equal(labels)
  }
  
  it should "achieve zero training error with loglikelihood" in {
    train(data, parser)
    for (x <- data) {
      assertPredictionCorrect(x, parser)
    }
  }
  
  it should "achieve zero training error with LaSO" in {
    trainLaso(data, parser)
    for (x <- data) {
      assertPredictionCorrect(x, parser)
    }
  }
  
  it should "distinguish unked entities" in {
    val model2 = PnpModel.init(true)
    val config2 = new SemanticParserConfig()
    config2.attentionCopyEntities = true
    config2.entityLinkingLearnedSimilarity = true
    config2.distinctUnkVectors = true
    val parser2 = SemanticParser.create(lexicon, vocab, config2, model2)
    
    val entities = List(("foo", "foo:e"), ("bar", "bar:e"), ("baz", "baz:e"))
    val trainingData = List(("foo", entities, "foo:e"),
        ("bar", entities, "bar:e"),
        ("city in bar", entities, "(lambda ($0) (and:<t*,t> (city:<e,t> $0) (loc:<e,<e,t>> $0 bar:e)))"),
        ("city in foo", entities, "(lambda ($0) (and:<t*,t> (city:<e,t> $0) (loc:<e,<e,t>> $0 foo:e)))"))

    val testData = List(("baz", entities, "baz:e"),
        ("city in baz", entities, "(lambda ($0) (and:<t*,t> (city:<e,t> $0) (loc:<e,<e,t>> $0 baz:e)))"))
    val trainingExamples = trainingData.map(x => encodeExample(x._1, x._2, x._3)) ++ data  
    val testExamples = testData.map(x => encodeExample(x._1, x._2, x._3))
    
    train(trainingExamples, parser2)
    
    for (x <- trainingExamples) {
      assertPredictionCorrect(x, parser2)
    }

    for (x <- testExamples) {
      assertPredictionCorrect(x, parser2)
    }
  }
}

case class SemanticParserExample(tokenIds: Array[Int], entityLinking: EntityLinking, lf: Expression2)