package org.allenai.wikitables;

import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Iterator;

import edu.stanford.nlp.sempre.Formula;
import edu.stanford.nlp.sempre.Builder;
import edu.stanford.nlp.sempre.Derivation;
import edu.stanford.nlp.sempre.Value;
import edu.stanford.nlp.sempre.ListValue;
import edu.stanford.nlp.sempre.ParserState;
import edu.stanford.nlp.sempre.Parser;
import edu.stanford.nlp.sempre.NumberFn;
import edu.stanford.nlp.sempre.JoinFn;
import edu.stanford.nlp.sempre.TypeInference;
import edu.stanford.nlp.sempre.TargetValuePreprocessor;
import edu.stanford.nlp.sempre.FloatingParser;
import edu.stanford.nlp.sempre.DerivationPruner;
import edu.stanford.nlp.sempre.Grammar;
import edu.stanford.nlp.sempre.LanguageAnalyzer;
import edu.stanford.nlp.sempre.corenlp.CoreNLPAnalyzer;
import edu.stanford.nlp.sempre.FuzzyMatchFn.FuzzyMatchFnMode;
import edu.stanford.nlp.sempre.tables.TableKnowledgeGraph;
import edu.stanford.nlp.sempre.tables.StringNormalizationUtils;
import edu.stanford.nlp.sempre.tables.dpd.DPDParser;
import edu.stanford.nlp.sempre.tables.test.*;
import edu.stanford.nlp.sempre.tables.match.*;
import fig.basic.*;

public class WikiTablesDataProcessor {
  public static List<CustomExample> getDataset(String path, boolean inSempreFormat,
                                               boolean includeDerivations, int beamSize) {
    CustomExample.opts.allowNoAnnotation = true;
    TableKnowledgeGraph.opts.baseCSVDir = "data/WikiTableQuestions";
    LanguageAnalyzer.opts.languageAnalyzer = "corenlp.CoreNLPAnalyzer";
    CoreNLPAnalyzer.opts.annotators = Arrays.asList(new String[] {"tokenize", "ssplit", "pos", "lemma", "ner"});
    EditDistanceFuzzyMatcher.opts.expandAbbreviations = true;
    EditDistanceFuzzyMatcher.opts.fuzzyMatchSubstring = true;
    EditDistanceFuzzyMatcher.opts.alsoReturnUnion = true;
    EditDistanceFuzzyMatcher.opts.alsoMatchPart = true;
    EditDistanceFuzzyMatcher.opts.fuzzyMatchMaxEditDistanceRatio = 0.3;
    if (inSempreFormat) {
        List<Pair<String, String>> pairedPaths = new ArrayList<Pair<String, String>>();
        pairedPaths.add(new Pair("train", path));
        List<CustomExample> dataset = CustomExample.getDataset(pairedPaths, null);
        if (includeDerivations)
            addDerivations(dataset, beamSize);
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
        if (includeDerivations)
            addDerivations(dataset, beamSize);
        return dataset;
    }
  }

  public static List<Pair<Pair<Integer, Integer>, Formula>> getEntityLinking(CustomExample ex) {
    // TODO: Return and EntityLinking object.
    List<Pair<Pair<Integer, Integer>, Formula>> entityLinking = new ArrayList<>();
    List<String> exTokens = ex.getTokens();
    EditDistanceFuzzyMatcher.opts.expandAbbreviations = true;
    EditDistanceFuzzyMatcher.opts.fuzzyMatchSubstring = true;
    EditDistanceFuzzyMatcher.opts.alsoReturnUnion = true;
    EditDistanceFuzzyMatcher.opts.alsoMatchPart = true;
    EditDistanceFuzzyMatcher.opts.fuzzyMatchMaxEditDistanceRatio = 0.3;
    
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

  static void addDerivations(List<CustomExample> dataset, int beamSize) {
    // Parses the examples in the given dataset, and stores all the correct derivations.

    // Setting all the options typically selected by Sempre
    // TODO: Make these actual command line arguments.
    Builder.opts.parser = "tables.dpd.DPDParser";
    DPDParser.opts.cheat = true;
    Builder.opts.executor = "tables.lambdadcs.LambdaDCSExecutor";
    Builder.opts.valueEvaluator = "tables.TableValueEvaluator";
    Parser.opts.beamSize = beamSize;
    TargetValuePreprocessor.opts.targetValuePreprocessor = "tables.TableValuePreprocessor";
    StringNormalizationUtils.opts.numberCanStartAnywhere = true;
    StringNormalizationUtils.opts.num2CanStartAnywhere = true;
    NumberFn.opts.unitless = true;
    NumberFn.opts.alsoTestByConversion = true;
    NumberFn.opts.alsoTestByIsolatedNER = true;
    JoinFn.opts.specializedTypeCheck = false;
    JoinFn.opts.typeInference = true;
    TypeInference.opts.typeLookup = "tables.TableTypeLookup";
    FloatingParser.opts.useSizeInsteadOfDepth = true;
    FloatingParser.opts.maxDepth = 8;
    FloatingParser.opts.useAnchorsOnce = false;
    FloatingParser.opts.useMaxAnchors = 2;
    DerivationPruner.opts.pruningStrategies = Arrays.asList(new String[] {"emptyDenotation", "nonLambdaError", "atomic", "tooManyValues", "badSummarizerHead", "mistypedMerge", "doubleNext", "doubleSummarizers", "sameMerge", "unsortedMerge", "typeRowMerge"});
    DerivationPruner.opts.pruningComputers = Arrays.asList(new String[] {"tables.TableDerivationPruningComputer"});
    DerivationPruner.opts.recursivePruning = false;
    Grammar.opts.inPaths = Arrays.asList(new String[] {"data/grow.grammar"});
    Grammar.opts.binarizeRules = false;
    Grammar.opts.tags = Arrays.asList(new String[] {"scoped", "merge-and", "arithmetic", "comparison", "alternative", "neq", "yearrange", "part", "closedclass", "scoped-2args-merge-and"});
    // End of command line arguments.
    Builder builder = new Builder();
    builder.build();
    int maxNumFormulas = 0;
    int minNumFormulas = 100;
    int totalNumFormulas = 0;
    for (CustomExample ex: dataset) {
      List<Formula> correctFormulas = new ArrayList<>();
      ParserState state = builder.parser.parse(builder.params, ex, false);
      for (Derivation deriv: state.predDerivations) {
        Value pred = builder.executor.execute(deriv.formula, ex.context).value;
        if (pred instanceof ListValue)
          pred = ((TableKnowledgeGraph) ex.context.graph).getListValueWithOriginalStrings((ListValue) pred);
        double result = builder.valueEvaluator.getCompatibility(ex.targetValue, pred);
        if (result == 1) {
          correctFormulas.add(deriv.formula);
       }
      }
      ex.alternativeFormulas = correctFormulas;
      int numFormulas = correctFormulas.size();
      if (numFormulas > maxNumFormulas)
        maxNumFormulas = numFormulas;
      if (numFormulas < minNumFormulas)
        minNumFormulas = numFormulas;
      totalNumFormulas += numFormulas;
    }
    System.out.println("Finished adding derivations to the dataset.");
    System.out.println("Max number of formulas per question: " + maxNumFormulas);
    System.out.println("Min number of formulas per question: " + minNumFormulas);
    System.out.println("Average number of formulas per question: " + (float) totalNumFormulas / dataset.size());
  } 

  public static void main(String[] args) {
    String path = "data/WikiTableQuestions/data/training-before300.examples";
    //String path = "data/wikitables/wikitables_data.ldcs";
    List<CustomExample> dataset = WikiTablesDataProcessor.getDataset(path, true, true, 10);
    for (int i = 0; i < dataset.size(); i++) {
      CustomExample ex = dataset.get(i);
      System.out.println("Utterance: " + ex.utterance);
      System.out.println("Formula: " + ex.targetFormula);
      System.out.println("Answer: " + ex.targetValue);
      List<Pair<Pair<Integer, Integer>, Formula>> entityLinking = WikiTablesDataProcessor.getEntityLinking(ex);
      for (Pair<Pair<Integer, Integer>, Formula> p: entityLinking) {
        System.out.println("Entity: " + p.getFirst().getFirst() + " " + p.getFirst().getSecond() + " " + p.getSecond());
      }
      int numFormulas = ex.alternativeFormulas.size();
      for (Formula formula: ex.alternativeFormulas) {
        System.out.println("Correct formula: " + formula);
      }
    }
  } 
}
