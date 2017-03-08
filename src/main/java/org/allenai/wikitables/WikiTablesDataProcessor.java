package org.allenai.wikitables;

import java.io.*;
import java.util.*;
import java.util.zip.GZIPInputStream;

import edu.stanford.nlp.sempre.Formula;
import edu.stanford.nlp.sempre.Builder;
import edu.stanford.nlp.sempre.Derivation;
import edu.stanford.nlp.sempre.Value;
import edu.stanford.nlp.sempre.ListValue;
import edu.stanford.nlp.sempre.ParserState;
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
                                               boolean includeDerivations, String derivationsPath,
                                               int beamSize) {
    CustomExample.opts.allowNoAnnotation = true;
    TableKnowledgeGraph.opts.baseCSVDir = "data/WikiTableQuestions";
    LanguageAnalyzer.opts.languageAnalyzer = "corenlp.CoreNLPAnalyzer";
    CoreNLPAnalyzer.opts.annotators = Arrays.asList(new String[] {"tokenize", "ssplit", "pos", "lemma", "ner"});
    EditDistanceFuzzyMatcher.opts.expandAbbreviations = true;
    EditDistanceFuzzyMatcher.opts.fuzzyMatchSubstring = true;
    EditDistanceFuzzyMatcher.opts.alsoReturnUnion = true;
    EditDistanceFuzzyMatcher.opts.alsoMatchPart = true;
    EditDistanceFuzzyMatcher.opts.fuzzyMatchMaxEditDistanceRatio = 0.3;
    List<CustomExample> dataset;
    if (inSempreFormat) {
      List<Pair<String, String>> pairedPaths = new ArrayList<>();
      pairedPaths.add(new Pair("train", path));
      dataset = CustomExample.getDataset(pairedPaths, null);
    } else {
      dataset = new ArrayList<>();
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
    }
    if (includeDerivations) {
      if (derivationsPath == null)
        computeDerivations(dataset, beamSize);
      else
        addDerivations(dataset, derivationsPath);
      int maxNumFormulas = 0;
      int minNumFormulas = (int) Double.POSITIVE_INFINITY;
      int totalNumFormulas = 0;
      int numZeroFormulas = 0;
      for (CustomExample ex: dataset) {
        int numFormulas = ex.alternativeFormulas.size();
        if (numFormulas > maxNumFormulas)
          maxNumFormulas = numFormulas;
        if (numFormulas < minNumFormulas)
          minNumFormulas = numFormulas;
        numZeroFormulas += numFormulas == 0 ? 1 : 0;
        totalNumFormulas += numFormulas;
      }
      System.out.println("Finished adding derivations to the dataset.");
      System.out.println("Max number of formulas per question: " + maxNumFormulas);
      System.out.println("Min number of formulas per question: " + minNumFormulas);
      System.out.println("Average number of formulas per question: " + (float) totalNumFormulas / dataset.size());
      System.out.println(numZeroFormulas + " questions did not yield any logical forms.");
    }
    return dataset;
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
    Set<Formula> formulasPresent = new HashSet<>();
    for (int i=0; i <= exTokens.size()-1; i++) {
      for (int j=i+1; j <= exTokens.size(); j++) {
        if (j == i+1) {
          // We're looking at a span of one token
          String token = exTokens.get(i);
          if (token.matches("[-+]?\\d*\\.?\\d+"))
            //entityLinking.add(new Pair(new Pair(i, j), Formula.fromString("(number "+token+")")));
            entityLinking.add(new Pair(new Pair(i, j), Formula.fromString(token)));
        }
        Collection<Formula> linkedFormulas = matcher.getFuzzyMatchedFormulas(exTokens, i, j,
                                         FuzzyMatchFnMode.ENTITY);
        linkedFormulas.addAll(matcher.getFuzzyMatchedFormulas(exTokens, i, j,
                FuzzyMatchFnMode.BINARY));
        for (Formula formula: linkedFormulas) {
          entityLinking.add(new Pair(new Pair(i, j), formula));
          formulasPresent.add(formula);
        }
      }
    }
    // This gives all formulas that can be yielded by the table
    Collection<Formula> unlinkedFormulas = matcher.getAllFormulas(FuzzyMatchFnMode.ENTITY);
    //unlinkedFormulas.addAll(matcher.getAllFormulas(FuzzyMatchFnMode.UNARY));
    unlinkedFormulas.addAll(matcher.getAllFormulas(FuzzyMatchFnMode.BINARY));
    // Adding unlinked entities with a null span
    for (Formula formula: unlinkedFormulas) {
      if (! formulasPresent.contains(formula))
        entityLinking.add(new Pair(null, formula));
    }
    // Sempre often generates formulas that contain 1, 0 and -1. Adding them as unlinked entities.
    //entityLinking.add(new Pair(null, Formula.fromString("(number 0)")));
    //entityLinking.add(new Pair(null, Formula.fromString("(number 1)")));
    //entityLinking.add(new Pair(null, Formula.fromString("(number -1)")));
    entityLinking.add(new Pair(null, Formula.fromString("0")));
    entityLinking.add(new Pair(null, Formula.fromString("1")));
    entityLinking.add(new Pair(null, Formula.fromString("-1")));
    return entityLinking;
  }

  static void addDerivations(List<CustomExample> dataset, String derivationsPath) {
    for (CustomExample ex: dataset) {
      String exId = ex.getId();
      File file = new File(derivationsPath + "/" + exId + ".gz");
      if (file.exists()) {
        try {
          BufferedReader reader = new BufferedReader(new InputStreamReader(
                  new GZIPInputStream(new FileInputStream(file))));
          List<Formula> correctFormulas = new ArrayList<>();
          String line;
          // TODO: Sort the derivations by length and set a limit on their number.
          while ((line = reader.readLine()) != null) {
            correctFormulas.add(Formula.fromString(line));
          }
          ex.alternativeFormulas = correctFormulas;
        } catch (IOException e) {
          e.printStackTrace();
        }
      }
    }
  }

  static void computeDerivations(List<CustomExample> dataset, int beamSize) {
    // Parses the examples in the given dataset, and stores all the correct derivations.

    // Setting all the options typically selected by Sempre
    // TODO: Make these actual command line arguments.
    Builder.opts.parser = "tables.dpd.DPDParser";
    DPDParser.opts.cheat = true;
    DPDParser.opts.dpdParserBeamSize = beamSize;
    Builder.opts.executor = "tables.lambdadcs.LambdaDCSExecutor";
    Builder.opts.valueEvaluator = "tables.TableValueEvaluator";
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
    float sumAvgNumCorrect = 0;
    int numOffBeam = 0;
    for (CustomExample ex: dataset) {
      int numAllDerivations = 0;
      List<Formula> correctFormulas = new ArrayList<>();
      ParserState state = builder.parser.parse(builder.params, ex, false);
      if (state.fallOffBeam)
        numOffBeam += 1;
      for (Derivation deriv: state.predDerivations) {
        numAllDerivations += 1;
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
      sumAvgNumCorrect += numAllDerivations == 0 ? 0.0 : (float) numFormulas / numAllDerivations;
    }
    System.out.println("Average proportion of correct formulas per question: " + sumAvgNumCorrect / dataset.size());
    System.out.println(numOffBeam + " out of " + dataset.size() + " questions had derivations that fell off the beam.");
  }

  public static void main(String[] args) {
    //String path = "data/wikitables/wikitables_sample_small.examples";
    String derivationsPath = "data/WikiTableQuestions/all_lfs";
    String path = "data/WikiTableQuestions/data/training-before300.examples";
    List<CustomExample> dataset = WikiTablesDataProcessor.getDataset(path, true, true, derivationsPath, 50);
    for (int i = 0; i < dataset.size(); i++) {
      CustomExample ex = dataset.get(i);
      System.out.println("Utterance: " + ex.utterance);
      System.out.println("Formula: " + ex.targetFormula);
      System.out.println("Answer: " + ex.targetValue);
      List<Pair<Pair<Integer, Integer>, Formula>> entityLinking = WikiTablesDataProcessor.getEntityLinking(ex);
      for (Pair<Pair<Integer, Integer>, Formula> p: entityLinking) {
        if (p.getFirst() == null)
          System.out.println("Unlinked entity: " + p.getSecond());
        else
          System.out.println("Linked entity: " + p.getFirst().getFirst() + " " + p.getFirst().getSecond() + " " + p.getSecond());
      }
    }
  }
}