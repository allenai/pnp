package org.allenai.wikitables;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.zip.GZIPInputStream;

import com.google.common.collect.Lists;
import com.jayantkrish.jklol.ccg.lambda.ExpressionParser;
import com.jayantkrish.jklol.ccg.lambda2.Expression2;

import edu.stanford.nlp.sempre.Builder;
import edu.stanford.nlp.sempre.ContextValue;
import edu.stanford.nlp.sempre.Derivation;
import edu.stanford.nlp.sempre.DerivationPruner;
import edu.stanford.nlp.sempre.FloatingParser;
import edu.stanford.nlp.sempre.Formula;
import edu.stanford.nlp.sempre.FuzzyMatchFn.FuzzyMatchFnMode;
import edu.stanford.nlp.sempre.Grammar;
import edu.stanford.nlp.sempre.JoinFn;
import edu.stanford.nlp.sempre.ListValue;
import edu.stanford.nlp.sempre.NumberFn;
import edu.stanford.nlp.sempre.ParserState;
import edu.stanford.nlp.sempre.TargetValuePreprocessor;
import edu.stanford.nlp.sempre.TypeInference;
import edu.stanford.nlp.sempre.Value;
import edu.stanford.nlp.sempre.corenlp.CoreNLPAnalyzer;
import edu.stanford.nlp.sempre.tables.StringNormalizationUtils;
import edu.stanford.nlp.sempre.tables.TableKnowledgeGraph;
import edu.stanford.nlp.sempre.tables.TableValueEvaluator;
import edu.stanford.nlp.sempre.tables.TableValuePreprocessor;
import edu.stanford.nlp.sempre.tables.dpd.DPDParser;
import edu.stanford.nlp.sempre.tables.lambdadcs.LambdaDCSExecutor;
import edu.stanford.nlp.sempre.tables.match.EditDistanceFuzzyMatcher;
import edu.stanford.nlp.sempre.tables.match.FuzzyMatcher;
import edu.stanford.nlp.sempre.tables.test.CustomExample;
import fig.basic.LispTree;
import fig.basic.Pair;

public class WikiTablesDataProcessor {
  
  private static Builder SEMPRE_BUILDER = getSempreBuilder(100);
  
  public static List<CustomExample> getDataset(String path, boolean inSempreFormat,
                                               boolean includeDerivations, String derivationsPath,
                                               int beamSize, int numDerivationsLimit) {
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
      if (derivationsPath == null) {
        computeDerivations(dataset, beamSize);
      } else {
        addDerivations(dataset, derivationsPath, numDerivationsLimit);
      }
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

  public static List<Pair<Pair<Integer, Integer>, Formula>> getEntityLinking(WikiTablesExample example) {
    List<Pair<Pair<Integer, Integer>, Formula>> entityLinking = new ArrayList<>();
    List<String> exTokens = example.sentence().getWords();
    EditDistanceFuzzyMatcher.opts.expandAbbreviations = true;
    EditDistanceFuzzyMatcher.opts.fuzzyMatchSubstring = true;
    EditDistanceFuzzyMatcher.opts.alsoReturnUnion = true;
    EditDistanceFuzzyMatcher.opts.alsoMatchPart = true;
    EditDistanceFuzzyMatcher.opts.fuzzyMatchMaxEditDistanceRatio = 0.3;
    
    FuzzyMatcher matcher = new EditDistanceFuzzyMatcher((TableKnowledgeGraph) example.getContext().graph);
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
    // This gives all formulas that can be yielded by the table
    Collection<Formula> unlinkedFormulas = matcher.getAllFormulas(FuzzyMatchFnMode.ENTITY);
    //unlinkedFormulas.addAll(matcher.getAllFormulas(FuzzyMatchFnMode.UNARY));
    unlinkedFormulas.addAll(matcher.getAllFormulas(FuzzyMatchFnMode.BINARY));
    // Adding unlinked entities with a null span
    for (Formula formula: unlinkedFormulas) {
      if (! formulasPresent.containsKey(formula)) {
        entityLinking.add(new Pair(null, formula));
        formulasPresent.put(formula, null);
      }
    }
    // Sempre often generates formulas that contain 1, 0 and -1. Adding them as unlinked entities.
    entityLinking.add(new Pair(null, Formula.fromString("0")));
    entityLinking.add(new Pair(null, Formula.fromString("1")));
    entityLinking.add(new Pair(null, Formula.fromString("-1")));
    return entityLinking;
  }

  static void addDerivations(List<CustomExample> dataset, String derivationsPath,
      int numDerivationsLimit) {
    if (numDerivationsLimit != -1) {
      System.out.println("Limiting number of derivations per example to " + numDerivationsLimit);
    }

    for (CustomExample ex: dataset) {
      String exId = ex.getId();
      File file = new File(derivationsPath + "/" + exId + ".gz");

      List<Formula> correctFormulas = Lists.newArrayList();
      if (file.exists()) {
        try {
          BufferedReader reader = new BufferedReader(new InputStreamReader(
              new GZIPInputStream(new FileInputStream(file))));
          String line;
          while ((line = reader.readLine()) != null) {
            correctFormulas.add(Formula.fromString(line));
          }
        } catch (IOException e) {
          e.printStackTrace();
        }
      }
      
      if (numDerivationsLimit >= 0 && correctFormulas.size() > numDerivationsLimit) {        
        List<Pair<Integer, Formula>> formulasWithSizes = Lists.newArrayList();
        for (Formula f : correctFormulas) {
          Expression2 e = ExpressionParser.expression2().parse(f.toString());
          formulasWithSizes.add(Pair.newPair(e.size(), f));
        }

        formulasWithSizes.sort(new DerivationLengthComparator());
        correctFormulas.clear();
        for (Pair<Integer, Formula> p : formulasWithSizes.subList(0, numDerivationsLimit)) {
          correctFormulas.add(p.getSecond());
        }
      }

      ex.alternativeFormulas = correctFormulas;
    }
  }

  static Builder getSempreBuilder(int beamSize) {
    // Setting all the options typically selected by Sempre
    // TODO: Make these actual command line arguments.
    Builder.opts.parser = "tables.dpd.DPDParser";
    DPDParser.opts.cheat = true;
    DPDParser.opts.dpdParserBeamSize = beamSize;
    Builder.opts.executor = "tables.lambdadcs.LambdaDCSExecutor";
    Builder.opts.valueEvaluator = "tables.TableValueEvaluator";
    LambdaDCSExecutor.opts.genericDateValue = true;
    TableValueEvaluator.opts.allowMismatchedTypes = true;
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
    return builder;
  }

  static void computeDerivations(List<CustomExample> dataset, int beamSize) {
    // Parses the examples in the given dataset, and stores all the correct derivations.
    Builder builder = getSempreBuilder(beamSize);
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
        if (isFormulaCorrect(deriv.formula, ex.context, ex.targetValue, builder)) {
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

  public static boolean isFormulaCorrect(Formula formula, ContextValue context, Value targetValue,
                                         Builder builder) {
    if (builder == null) {
      builder = SEMPRE_BUILDER;
    }

    Value pred = builder.executor.execute(formula, context).value;
    if (pred instanceof ListValue)
      pred = ((TableKnowledgeGraph) context.graph).getListValueWithOriginalStrings((ListValue) pred);
    
    TableValuePreprocessor targetPreprocessor = new TableValuePreprocessor();
    targetValue = targetPreprocessor.preprocess(targetValue);
    // Print the predicted and target values.
    // System.out.println(pred + " " + targetValue);

    double result = builder.valueEvaluator.getCompatibility(targetValue, pred);
    return result == 1;
  }
}

class DerivationLengthComparator implements Comparator<Pair<Integer, Formula>> {

  @Override
  public int compare(Pair<Integer, Formula> o1, Pair<Integer, Formula> o2) {
    return Integer.compare(o1.getFirst(), o2.getFirst());
  }
}
