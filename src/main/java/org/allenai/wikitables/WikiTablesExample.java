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
    System.out.println("Before conversion: " + pnpFormula.toString());
    String expressionString = toSempreLambdaForm(pnpFormula.toString());
    System.out.println("After conversion: " + expressionString);
    try {
      Formula sempreFormula = Formula.fromString(expressionString);
      return WikiTablesDataProcessor.isFormulaCorrect(sempreFormula, context, targetValue, null);
    } catch (Exception ex) {
      System.err.println("Bad formula: " + expressionString);
      return false;
    }
  }

  public String toString() {
    List<String> logicalFormStrings = new ArrayList<>();
    for (Expression2 lf : logicalForms) {
        logicalFormStrings.add(lf.toString());
    }
    return sentence + " [\n" + String.join(" ", logicalFormStrings) + "\n]";
  }

  public static String toPnpLambdaForm(String expression) {
    /*
    Sempre's lambda expressions are written differently from what pnp expects. We make the following change
    (lambda x ((reverse fb:cell.cell.number) (var x))) -> (lambda (x) ((reverse fb:cell.cell.number) x))
     */
    // Sempre's lambda expressions have only one free variable, I think.
    expression = expression.replaceAll("lambda x", "lambda (x)");
    expression = expression.replaceAll("(var x)", "x");
    return expression;
  }

  public static String toSempreLambdaForm(String expression) {
    // TODO: Properly de-canonicalize variable names, to work with multiple variables.
    expression = expression.replaceAll("lambda \\(\\$0\\)", "lambda x");
    // Remove single sub-expressions within parens. Eg: (fb:type.object.type (fb:type.row)) -> (fb:type.object.type fb:type.row)
    expression = expression.replaceAll("\\(([^ ]*)\\)", "$1");
    expression = expression.replaceAll("\\$0", "(var x)");
    return expression;
  }

  public static void main(String[] args) {
    System.out.println(toSempreLambdaForm("((reverse fb:cell.cell.number) ((reverse fb:row.row.round_2) (fb:type.object.type (fb:type.row))))"));
  }
}
