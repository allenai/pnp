package org.allenai.pnp.examples

import scala.collection.mutable.ListBuffer
import scala.collection.mutable.MultiMap
import scala.collection.mutable.{ HashMap, Set }

import com.google.common.collect.Multimap
import com.jayantkrish.jklol.util.IndexedList
import com.jayantkrish.jklol.models.VariableNumMap
import com.jayantkrish.jklol.models.DiscreteVariable
import org.allenai.pnp.ParametricPpModel
import org.allenai.pnp.CompGraphNode
import org.allenai.pnp.Pp

// Classes for representing CFG parse trees
abstract class Parse(val pos: String) {
  def getTokens: List[String]
}

case class Terminal(word: String, override val pos: String) extends Parse(pos) {
  override def getTokens: List[String] = {
    List(word)
  }

  override def toString: String = {
    pos + " -> " + word
  }
}

case class Nonterminal(left: Parse, right: Parse, override val pos: String) extends Parse(pos) {
  override def getTokens: List[String] = {
    left.getTokens ++ right.getTokens
  }

  override def toString: String = {
    pos + " -> (" + left.toString + ", " + right.toString + ")"
  }
}

// Classes for representing actions of a shift/reduce
// parser
abstract class Action {
  def apply(state: ShiftReduceState): ShiftReduceState
  def score(state: ShiftReduceState): Pp[CompGraphNode]
  def addParams(names: IndexedList[String], vars: ListBuffer[VariableNumMap]): Unit
}

class Shift(val word: String, val pos: String) extends Action {
  val terminal = Terminal(word, pos)
  val paramName = "shift_" + word + "_" + pos

  override def apply(state: ShiftReduceState): ShiftReduceState = {
    ShiftReduceState(state.tokens.tail, (terminal :: state.stack))
  }

  override def score(state: ShiftReduceState): Pp[CompGraphNode] = {
    Pp.param(paramName)
  }

  override def addParams(names: IndexedList[String], vars: ListBuffer[VariableNumMap]): Unit = {
    names.add(paramName)
    vars += VariableNumMap.EMPTY
  }
}

class Reduce(val leftPos: String, val rightPos: String, val rootPos: String) extends Action {
  val paramName = "reduce_" + leftPos + "_" + rightPos + "_" + rootPos

  override def apply(state: ShiftReduceState): ShiftReduceState = {
    val left = state.stack(1)
    val right = state.stack(0)
    // TODO: preconditions check
    val nonterminal = Nonterminal(left, right, rootPos)
    ShiftReduceState(state.tokens, (nonterminal :: state.stack.drop(2)))
  }

  override def score(state: ShiftReduceState): Pp[CompGraphNode] = {
    Pp.param(paramName)
  }

  override def addParams(names: IndexedList[String], vars: ListBuffer[VariableNumMap]): Unit = {
    names.add(paramName)
    vars += VariableNumMap.EMPTY
  }
}

// A ShiftReduceState represents the state of a shift
// reduce parser. It consists of a queue of unconsumed
// tokens and a stack of already-built parses.
case class ShiftReduceState(tokens: List[String], stack: List[Parse])

class PpParser(
    lexActions: MultiMap[String, Action],
    grammarActions: MultiMap[(String, String), Action]
) {

  def parse(sent: List[String]): Pp[Parse] = {
    parse(ShiftReduceState(sent, List()))
  }

  def parse(state: ShiftReduceState): Pp[Parse] = {
    val tokens = state.tokens
    val stack = state.stack
    if (tokens.size == 0 && stack.size == 1) {
      // All tokens consumed and all possible
      // reduces performed.
      Pp.value(stack.head)
    } else {
      // Queue for each possible action
      val actions = ListBuffer[Action]()

      // Queue shift actions
      if (tokens.size > 0) {
        val shifts = lexActions.getOrElse(tokens.head, Set())
        actions ++= shifts
      }

      // Queue reduce actions
      if (stack.size >= 2) {
        val left = stack(1)
        val right = stack(0)
        val reduces = grammarActions.getOrElse((left.pos, right.pos), Set())
        actions ++= reduces
      }

      for {
        // Score each possible action, nondeterministically
        // select one to apply, then recurse on the next
        // parser state.
        scores <- scoreActions(state, actions);
        action <- Pp.choose(actions.toArray, scores)
        p <- parse(action.apply(state))
      } yield {
        p
      }
    }
  }

  def scoreActions(state: ShiftReduceState, actions: ListBuffer[Action]): Pp[Array[CompGraphNode]] = {
    val scoreList = actions.foldRight(Pp.value(List[CompGraphNode]()))((action, list) =>
      for {
        x <- action.score(state);
        l <- list
      } yield {
        x :: l
      })

    scoreList.flatMap { x => Pp.value(x.toArray) }
  }

  def getParams: ParametricPpModel = {
    val paramNames = IndexedList.create[String]()
    val paramVars = ListBuffer[VariableNumMap]()

    lexActions.values.foreach(_.foreach(_.addParams(paramNames, paramVars)))
    grammarActions.values.foreach(_.foreach(_.addParams(paramNames, paramVars)))

    new ParametricPpModel(paramNames, paramVars.toList)
  }
}

object PpParser {

  def fromMaps(
    lexicon: List[(String, String)],
    grammar: List[((String, String), String)]
  ): PpParser = {

    val lexActions = new HashMap[String, Set[Action]] with MultiMap[String, Action]
    for ((k, v) <- lexicon) {
      lexActions.addBinding(k, new Shift(k, v))
    }

    val grammarActions = new HashMap[(String, String), Set[Action]] with MultiMap[(String, String), Action]
    for ((k, v) <- grammar) {
      grammarActions.addBinding(k, new Reduce(k._1, k._2, v))
    }

    new PpParser(lexActions, grammarActions)
  }
}

