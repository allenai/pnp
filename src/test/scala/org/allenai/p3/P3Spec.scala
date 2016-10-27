package org.allenai.p3

import java.util.Collections

import scala.collection.JavaConverters._

import org.allenai.pnp.Env
import org.allenai.pnp.ParametricPpModel
import org.allenai.pnp.Pp
import org.allenai.pnp.PpModel
import org.allenai.pnp.PpUtil
import org.scalatest._

import com.jayantkrish.jklol.ccg.CcgCkyInference
import com.jayantkrish.jklol.ccg.CcgParser
import com.jayantkrish.jklol.ccg.DefaultCcgFeatureFactory
import com.jayantkrish.jklol.ccg.ParametricCcgParser
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.ccg.lambda2.ExpressionSimplifier
import com.jayantkrish.jklol.models.DiscreteVariable
import com.jayantkrish.jklol.models.VariableNumMap
import com.jayantkrish.jklol.nlpannotation.AnnotatedSentence
import com.jayantkrish.jklol.parallel.LocalMapReduceExecutor
import com.jayantkrish.jklol.training.Lbfgs
import com.jayantkrish.jklol.training.LbfgsConvergenceError
import com.jayantkrish.jklol.training.NullLogFunction
import com.jayantkrish.jklol.util.IndexedList
import com.jayantkrish.jklol.parallel.MapReduceConfiguration

class P3Spec extends FlatSpec with Matchers {

  val TOLERANCE = 0.0001

  val lexicon = Array(
    "1,N{0},1,0 num",
    "2,N{0},2,0 num",
    "3,N{0},3,0 num",
    "4,N{0},4,0 num",
    "+,((N{1}\\N{1}){0}/N{2}){0},(lambda (x y) (+ x y)),0 +",
    "1_or_2,N{0},(amb (list 1 2) (list 2.0 3.0)),0 num",
    "plus,((N{1}\\N{1}){0}/N{2}){0},(lambda (x y) (+ x y)),0 plus",
    "plus,((N{1}\\N{1}){0}/N{2}){0},(lambda (x y) (* x y)),0 plus",
    "times,((N{1}\\N{1}){0}/N{2}){0},(lambda (x y) (+ x y)),0 times",
    "times,((N{1}\\N{1}){0}/N{2}){0},(lambda (x y) (* x y)),0 times",
    "equals,((S{0}\\N{1}){0}/N{2}){0},(lambda (x y) (= x y)),0 equals",
    "even,N{0},randeven,0 even",
    "odd,N{0},randodd,0 odd"
  )

  val nums = Array(1, 2)
  val randeven = for {
    param <- Pp.param("randeven")
    v <- Pp.choose(nums, param)
  } yield {
    v
  }

  val randodd = for {
    param <- Pp.param("randeven")
    v <- Pp.choose(nums, param)
  } yield {
    v
  }

  val bindings = Map[String, AnyRef](
    "1" -> 1.asInstanceOf[AnyRef],
    "2" -> 2.asInstanceOf[AnyRef], "3" -> 3.asInstanceOf[AnyRef],
    "4" -> 4.asInstanceOf[AnyRef],
    "2.0" -> 2.0.asInstanceOf[AnyRef], "3.0" -> 3.0.asInstanceOf[AnyRef],
    "+" -> PpUtil.wrap2((x: Int, y: Int) => x + y),
    "*" -> PpUtil.wrap2((x: Int, y: Int) => x * y),
    "=" -> PpUtil.wrap2((x: AnyRef, y: AnyRef) => x == y),
    "randeven" -> randeven, "randodd" -> randodd,
    "list" -> ((x: Vector[AnyRef]) => Pp.value(x.toList)),
    "amb" -> PpUtil.wrap((x: List[AnyRef], w: List[Double]) => Pp.choose(x.toSeq, w.toSeq))
  )

  // Create training examples
  val examples = Array(
    ("1 plus 1 equals 2", true),
    ("1 plus 1 equals 2", true),
    ("1 plus 1 equals 1", true),
    ("even", 2),
    ("even", 2),
    ("even", 1)
  )

  def makeTrainingExample(sentence: String, label: AnyRef): P3PpExample = {
    val tokens = sentence.split(" ")
    val annotatedSentence = new AnnotatedSentence(
      tokens.toList.asJava,
      Collections.nCopies(tokens.size, ParametricCcgParser.DEFAULT_POS_TAG)
    );
    new ValueP3PpExample(annotatedSentence, Env.init, null, null, label)
  }

  val trainingData = examples.map(x => makeTrainingExample(x._1, (x._2).asInstanceOf[AnyRef]))

  def lfToPp(lf: Expression2): Pp[AnyRef] = {
    PpUtil.lfToPp(lf, bindings)
  }

  val rules = Array("FOO{0} BAR{0}")
  val ccgFamily = ParametricCcgParser.parseFromLexicon(
    lexicon.toList.asJava,
    List[String]().asJava, rules.toList.asJava,
    new DefaultCcgFeatureFactory(false, false), null, false, null, false
  );
  val ccgParser = ccgFamily.getModelFromParameters(ccgFamily.getNewSufficientStatistics());

  val simplifier = ExpressionSimplifier.lambdaCalculus();

  val pp = new PpModel(IndexedList.create[String], List())
  val p3 = new P3PpModel(ccgParser, pp, lfToPp)
  val inf = new P3PpBeamInference(CcgCkyInference.getDefault(100), simplifier, 10, 100)

  def beamSearch(model: P3PpModel, words: String): List[P3PpParse[AnyRef]] = {
    val tokens = words.split(" ").toList
    val sentence = new AnnotatedSentence(
      tokens.asJava,
      Collections.nCopies(tokens.size, ParametricCcgParser.DEFAULT_POS_TAG)
    );

    inf.beamSearch(model, sentence, Env.init, new NullLogFunction()).parses
  }

  "P3PpModel" should "perform inference correctly" in {
    val parses = beamSearch(p3, "1 + 2")

    parses.size should be(1)
    val parse = parses(0)
    parse.value should be(3)
    parse.prob should be(1.0)
  }

  it should "perform inference correctly (2)" in {
    val parses = beamSearch(p3, "1_or_2 + 2")

    parses.size should be(2)
    parses(0).value should be(4)
    parses(0).prob should be(3.0 +- TOLERANCE)
    parses(1).value should be(3)
    parses(1).prob should be(2.0 +- TOLERANCE)
  }

  it should "train" in {
    val intVarType = DiscreteVariable.sequence("ints", nums.size)
    val intVar = VariableNumMap.singleton(0, "ints", intVarType)
    val ppFamily = new ParametricPpModel(
      IndexedList.create(List("randeven", "randodd").asJava),
      List(intVar, intVar)
    )

    val p3Family = new ParametricP3PpModel(ccgFamily, ppFamily, lfToPp)

    MapReduceConfiguration.setMapReduceExecutor(new LocalMapReduceExecutor(1, 1))
    val oracle = new P3PpLoglikelihoodOracle(p3Family, inf)
    val trainer = new Lbfgs(10, 10, 0.0, new NullLogFunction())

    val initialParameters = oracle.initializeGradient()
    val parameters = try {
      trainer.train(oracle, initialParameters, trainingData.toList.asJava)
    } catch {
      case e: LbfgsConvergenceError => e.getFinalParameters
    }
    val model = p3Family.getModelFromParameters(parameters)

    val parses = beamSearch(model, "even plus 1")
    val partitionFunction = parses.map(_.prob).sum

    parses.size should be(4)
    parses(0).value should be(3)
    parses(0).prob / partitionFunction should be((4.0 / 9.0) +- TOLERANCE)
  }
}