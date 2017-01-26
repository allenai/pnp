package org.allenai.wikitables;

import java.util.Collection;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Iterator;

import edu.stanford.nlp.sempre.Formula;
import edu.stanford.nlp.sempre.FuzzyMatchFn.FuzzyMatchFnMode;
import edu.stanford.nlp.sempre.tables.TableKnowledgeGraph;
import edu.stanford.nlp.sempre.tables.test.*;
import edu.stanford.nlp.sempre.tables.match.*;
import fig.basic.*;

public class WikiTablesDataProcessor {
  private ArrayList<Pair<String, String>> pairedPaths;

  public WikiTablesDataProcessor(Map<String, String> dataPaths) {
    // dataPaths contains a map of datatype strings to their paths.
    // Eg: {"train": "/path/to/train", "dev": "/path/to/dev"}
    pairedPaths = new ArrayList<Pair<String, String>>();
    Iterator it = dataPaths.entrySet().iterator();
    while (it.hasNext()) {
      Map.Entry entry = (Map.Entry)it.next();
      pairedPaths.add(new Pair(entry.getKey(), entry.getValue()));
    }
  }

  public WikiTablesDataProcessor(String path) {
    this(new HashMap<String, String>() {{ put("train", path); }});
  }

  public List<CustomExample> getDataset() {
    CustomExample.opts.allowNoAnnotation = true;
    TableKnowledgeGraph.opts.baseCSVDir = "data/WikiTableQuestions";
    List<CustomExample> dataset = CustomExample.getDataset(pairedPaths, null);
    return dataset;
  }

  public List<Pair<Pair<Integer, Integer>, Formula>> getEntityLinking(CustomExample ex) {
    List<Pair<Pair<Integer, Integer>, Formula>> entityLinking = new ArrayList<>();
    List<String> exTokens = ex.getTokens();
    EditDistanceFuzzyMatcher.opts.expandAbbreviations = true;
    FuzzyMatcher matcher = new EditDistanceFuzzyMatcher((TableKnowledgeGraph) ex.context.graph);
    for (int i=0; i <= exTokens.size()-2; i++) {
      for (int j=i+1; j <= exTokens.size()-1; j++) {
        Collection<Formula> formulas = matcher.getFuzzyMatchedFormulas(ex.getTokens(), i, j,
                                         FuzzyMatchFnMode.ENTITY);
        for (Formula formula: formulas)
          entityLinking.add(new Pair(new Pair(i, j), formula));
      }
    }
    return entityLinking;
  }

  public static void main(String[] args) {
    String path = "data/WikiTableQuestions/data/training-before300.examples";
    WikiTablesDataProcessor dataProcessor = new WikiTablesDataProcessor(path);
    CustomExample ex = dataProcessor.getDataset().get(0);
    List<Pair<Pair<Integer, Integer>, Formula>> entityLinking = dataProcessor.getEntityLinking(ex);
    for (Pair<Pair<Integer, Integer>, Formula> p: entityLinking) {
      System.out.println(p.getFirst().getFirst() + " " + p.getFirst().getSecond() + " " + p.getSecond());
    }
  } 
}
