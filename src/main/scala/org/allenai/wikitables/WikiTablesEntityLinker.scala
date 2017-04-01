package org.allenai.wikitables

import scala.collection.JavaConverters._

import java.util.Collection
import edu.stanford.nlp.sempre.tables.`match`.EditDistanceFuzzyMatcher
import edu.stanford.nlp.sempre.Formula
import edu.stanford.nlp.sempre.FuzzyMatchFn.FuzzyMatchFnMode
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.{Set => MutableSet}
import org.allenai.pnp.semparse.Span
import edu.stanford.nlp.sempre.tables.TableKnowledgeGraph
import scala.util.Try
import java.nio.file.Paths
import java.nio.file.Files
import com.google.common.base.Preconditions
import scala.io.Source

class WikiTablesEntityLinker {
  
  import WikiTablesEntityLinker._

  def loadDataset(filename: String,
      examples: Seq[WikiTablesExample]
  ): Vector[RawEntityLinking] = {
    val preprocessedFile = filename + PREPROCESSING_SUFFIX
    
    val entityLinkings = if (Files.exists(Paths.get(preprocessedFile))) {
      RawEntityLinking.fromJsonFile(preprocessedFile)
    } else {
      val linkings = examples.map(getEntityLinking)
      RawEntityLinking.toJsonFile(preprocessedFile, linkings)
      linkings
    }

    Preconditions.checkState(examples.size == entityLinkings.size,
        "Wrong number of entity linkings (%s) for examples (%s). Filename: %s",
        entityLinkings.size.toString, examples.size.toString, filename)

    entityLinkings.toVector
  }
  
  def getEntityLinking(example: WikiTablesExample): RawEntityLinking = {
    val links = ListBuffer[(Option[Span], Formula)]()
    val foundFormulas = MutableSet[Formula]()

    EditDistanceFuzzyMatcher.opts.expandAbbreviations = true;
    EditDistanceFuzzyMatcher.opts.fuzzyMatchSubstring = true;
    EditDistanceFuzzyMatcher.opts.alsoReturnUnion = true;
    EditDistanceFuzzyMatcher.opts.alsoMatchPart = true;
    EditDistanceFuzzyMatcher.opts.fuzzyMatchMaxEditDistanceRatio = 0.3;

    val graph = example.getContext().graph.asInstanceOf[TableKnowledgeGraph]
    val matcher = new EditDistanceFuzzyMatcher(graph);

    // We don't actually use the spans of formulas at the moment.
    /*
    HashMap<Formula, Pair<Integer, Integer>> formulasPresent = new HashMap<>();
    for (int i=0; i <= exTokens.size()-1; i++) {
      for (int j=i+1; j <= exTokens.size(); j++) {
        if (j == i+1) {
          // We're looking at a span of one token
          String token = exTokens.get(i);
          if (token.matches("[-+]?\\d*\\.?\\d+")) {
            Formula formula = Formula.fromString(token);
            Pair<Integer, Integer> span = new Pair<>(i, j);
            entityLinking.add(new Pair<>(span, formula));
            formulasPresent.put(formula, span);
          }
        }
        Collection<Formula> linkedFormulas = matcher.getFuzzyMatchedFormulas(exTokens, i, j,
                                         FuzzyMatchFnMode.ENTITY);
        linkedFormulas.addAll(matcher.getFuzzyMatchedFormulas(exTokens, i, j,
                FuzzyMatchFnMode.BINARY));
        for (Formula formula: linkedFormulas) {
          // TODO: Store all the spans in entity linking instead of just the shortest one.
          Pair<Integer, Integer> span = new Pair<>(i, j);
          if (formulasPresent.containsKey(formula)) {
            Pair<Integer, Integer> previousSpan = formulasPresent.get(formula);
            int previousSpanLength = previousSpan.getSecond() - previousSpan.getFirst();
            int currentSpanLength = span.getSecond() - span.getFirst();
            if (currentSpanLength < previousSpanLength) {
              entityLinking.remove(new Pair<>(previousSpan, formula));
              entityLinking.add(new Pair<>(span, formula));
              formulasPresent.put(formula, span);
            }
          } else {
            entityLinking.add(new Pair(span, formula));
            formulasPresent.put(formula, span);
          }
        }
      }
    }
    */

    val nerFormulas = getNerEntityLinking(example.sentence
        .getAnnotation(WikiTablesUtil.NER_ANNOTATION).asInstanceOf[List[List[String]]])
    links ++= nerFormulas
    foundFormulas ++= nerFormulas.map(_._2)
    
    val entityFormulas = matcher.getAllFormulas(FuzzyMatchFnMode.ENTITY).asScala
    // TODO: these are unused, not sure why.
    // val unaryFormulas = matcher.getAllFormulas(FuzzyMatchFnMode.UNARY).asScala
    val binaryFormulas = matcher.getAllFormulas(FuzzyMatchFnMode.BINARY).asScala;
    val numberFormulas = Seq(-1, 0, 1).map(x => Formula.fromString(x.toString))
    
    // This gives all formulas that can be yielded by the table
    val unlinkedFormulas = entityFormulas ++ binaryFormulas ++ numberFormulas

    // Adding unlinked entities with a null span
    for (formula <- unlinkedFormulas) {
      if (!foundFormulas.contains(formula)) {
        links += ((None, formula))
        foundFormulas += formula
      }
    }

    return RawEntityLinking(example.id, links.toList)
  }
  
  def getNerEntityLinking(ner: List[List[String]]): Seq[(Option[Span], Formula)] = {
    val formulaIndexMap = ListBuffer[(Formula, Int)]()
    for ((t,i) <- ner.zipWithIndex) {
      if (t.length > 0) {
        val tag = t(0)
        val value = t(1)
        
        if (value != null) {
          val formulas = tag match {
            // either null or 1.0, 6.0, etc.
            case "ORDINAL" | "NUMBER" | "PERCENT" | "DURATION" | "MONEY" => tryParseNumber(value)
            // Dates have 180X, 2002, PRESENT_REF, 2010-05, 2012-SU, XXXX-03-06, 2002/2012
            case "DATE" => tryParseDate(value)
            // T03:59
            case "TIME" => tryParseTime(value)
          
            case _ => Seq()
          }

          formulaIndexMap ++= formulas.map(x => (x, i))
        }
      }
    }
    
    // Associate a span with each formula.
    // This code assumes that any given formula only occurs in
    // a single span in the text.
    val allFormulas = formulaIndexMap.map(_._1).toSet
    val formulasWithSpans = allFormulas.map{ x =>
      val indexes = formulaIndexMap.filter(_._1 == x).map(_._2)
      val start = indexes.min
      val end = indexes.max
      (Some(Span(start, end)), x)
    }

    formulasWithSpans.toList
  }
}

object WikiTablesEntityLinker {
   
  val PREPROCESSING_SUFFIX = ".entities.json"
  
  def numberToFormula(n: Double): Formula = {
    if (n.toInt == n) {
      Formula.fromString(n.toInt.toString)
    } else {
      Formula.fromString(n.formatted("%.3f"))
    }
  }
 
  def tryParseNumber(s: String): Seq[Formula] = {
    val parts = s.split("[-/]")
    val values = for {
      p <- parts
      replaced = p.replaceAll("[^0-9E.]", "")
      o = Try(replaced.toDouble).toOption
      if o.isDefined
    } yield {
      o.get
    }

    values.map(numberToFormula)
  }
  
  def tryParseDate(s: String): Seq[Formula] = {
    val parts = s.split("[-/]")
    val values = for {
      p <- parts
      replaced = p.replaceAll("X", "0").replaceAll("[^0-9E.]", "")
      o = Try(replaced.toDouble).toOption
      if o.isDefined
    } yield {
      o.get
    }

    values.map(numberToFormula)
  }

  def tryParseTime(s: String): Seq[Formula] = {
    // Time references in WikiTables tend to be to cells
    // of the table, and therefore don't need to be linked
    // to special formulas. Leaving this hook here for future
    // use if necessary.
    Seq()
  } 
}