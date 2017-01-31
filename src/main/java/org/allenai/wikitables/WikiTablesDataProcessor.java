package org.allenai.wikitables;

import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.IOException;
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
  public static List<CustomExample> getDataset(String path, boolean inSempreFormat) {
    CustomExample.opts.allowNoAnnotation = true;
    TableKnowledgeGraph.opts.baseCSVDir = "data/WikiTableQuestions";
    if (inSempreFormat) {
        List<Pair<String, String>> pairedPaths = new ArrayList<Pair<String, String>>();
        pairedPaths.add(new Pair("train", path));
        List<CustomExample> dataset = CustomExample.getDataset(pairedPaths, null);
        return dataset;
    } else {
        List<CustomExample> dataset = new ArrayList<CustomExample>();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
            int exampleId = 0;
            String line;
            while ((line = reader.readLine()) != null) {
                String[] lineParts = line.split("\t");
                LispTree tree = LispTree.proto.parseFromString(lineParts[0]);
                CustomExample ex = CustomExample.fromLispTree(tree, Integer.toString(exampleId));
                // This does things like tokenizing the utterance.
                ex.preprocess();
                ex.targetFormula = Formula.fromString(lineParts[1]);
                dataset.add(ex);
                exampleId++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return dataset;
    }
  }

  public static List<Pair<Pair<Integer, Integer>, Formula>> getEntityLinking(CustomExample ex) {
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
    //String path = "data/WikiTableQuestions/data/training-before300.examples";
    String path = "data/wikitables/wikitables_data.ldcs";
    List<CustomExample> dataset = WikiTablesDataProcessor.getDataset(path, false);
    for (int i = 0; i < dataset.size(); i++) {
        CustomExample ex = dataset.get(i);
        System.out.println("Utterance: " + ex.utterance);
        System.out.println("Formula: " + ex.targetFormula);
        List<Pair<Pair<Integer, Integer>, Formula>> entityLinking = WikiTablesDataProcessor.getEntityLinking(ex);
        for (Pair<Pair<Integer, Integer>, Formula> p: entityLinking) {
        System.out.println("Entity: " + p.getFirst().getFirst() + " " + p.getFirst().getSecond() + " " + p.getSecond());
        }
    }
  } 
}
