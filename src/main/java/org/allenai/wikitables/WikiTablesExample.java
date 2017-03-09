package org.allenai.wikitables;

import java.util.Set;
import java.util.List;
import java.util.ArrayList;

import com.jayantkrish.jklol.ccg.lambda2.Expression2;
import com.jayantkrish.jklol.nlpannotation.AnnotatedSentence;
import edu.stanford.nlp.sempre.ContextValue;
import edu.stanford.nlp.sempre.Formula;
import edu.stanford.nlp.sempre.Value;

/**
 * Similar to CcgExample, except that this supports multiple logical forms per example, and also stores the
 * context and target values from Sempre to execute and evaluate formulas.
 * This currently does not hold a syntactic parse or dependencies. But we may need them in the future.
 */

public class WikiTablesExample {

  private final AnnotatedSentence sentence;
  private final Set<Expression2> logicalForms;
  private final ContextValue context;
  private final Value targetValue;

  public WikiTablesExample(AnnotatedSentence sentence, Set<Expression2> logicalForms, ContextValue context,
                           Value targetValue) {
    this.sentence = sentence;
    this.logicalForms = logicalForms;
    this.context = context;
    this.targetValue = targetValue;
  }

  public Set<Expression2> getLogicalForms() {
    return logicalForms;
  }

  public AnnotatedSentence getSentence() {
    return sentence;
  }

  public boolean isFormulaCorrect(Expression2 pnpFormula) {
    // Sempre represents lambda expressions differently. We changed them when reading the examples. Changing
    // them back for execution.
    String expressionString = pnpFormula.toString().replaceAll("lambda (x)", "lambda x");
    Formula sempreFormula = Formula.fromString(expressionString);
    return WikiTablesDataProcessor.isFormulaCorrect(sempreFormula, context, targetValue, null);
  }

  public String toString() {
    List<String> logicalFormStrings = new ArrayList<>();
    for (Expression2 lf : logicalForms) {
        logicalFormStrings.add(lf.toString());
    }
    return sentence + " [\n" + String.join(" ", logicalFormStrings) + "\n]";
  }
}
