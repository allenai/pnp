package org.allenai.wikitables;

import edu.stanford.nlp.sempre.Formula;

public class WikiTablesUtil {
    public static String toPnpLogicalForm(Formula expression) {
      /*
      Sempre's lambda expressions are written differently from what pnp expects. We make the following change
      (lambda x ((reverse fb:cell.cell.number) (var x))) -> (lambda (x) ((reverse fb:cell.cell.number) x))
       */
      // Sempre's lambda expressions have only one free variable, I think.
      String expressionString = expression.toString().replaceAll("lambda x", "lambda (x)");
      expressionString = expressionString.replaceAll("(var x)", "x");
      return expressionString;
    }

    public static String toSempreLogicalForm(String expression) {
      // TODO: Properly de-canonicalize variable names, to work with multiple variables.
      expression = expression.replaceAll("lambda \\(\\$0\\)", "lambda x");
      // Remove single sub-expressions within parens. Eg: (fb:type.object.type (fb:type.row)) -> (fb:type.object.type fb:type.row)
      expression = expression.replaceAll("\\(([^ ]*)\\)", "$1");
      expression = expression.replaceAll("\\$0", "(var x)");
      return expression;
    }
}
