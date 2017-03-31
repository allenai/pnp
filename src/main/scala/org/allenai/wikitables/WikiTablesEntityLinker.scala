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

class WikiTablesEntityLinker {
  
  /*
  def loadDataset(filename: String): Map[String, RawEntityLinking] = {
    
  }
  */
  
  def getEntityLinking(example: WikiTablesExample): RawEntityLinking = {
    val links = ListBuffer[(Option[Span], Formula)]()
    val foundFormulas = MutableSet[Formula]()

    EditDistanceFuzzyMatcher.opts.expandAbbreviations = true;
    EditDistanceFuzzyMatcher.opts.fuzzyMatchSubstring = true;
    EditDistanceFuzzyMatcher.opts.alsoReturnUnion = true;
    EditDistanceFuzzyMatcher.opts.alsoMatchPart = true;
    EditDistanceFuzzyMatcher.opts.fuzzyMatchMaxEditDistanceRatio = 0.3;

    val matcher = new EditDistanceFuzzyMatcher(example.getContext().graph
        .asInstanceOf[TableKnowledgeGraph]);

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

    return RawEntityLinking(links.toList)
  }
}