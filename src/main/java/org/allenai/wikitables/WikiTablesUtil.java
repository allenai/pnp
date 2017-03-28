package org.allenai.wikitables;

import java.util.*;

import com.jayantkrish.jklol.ccg.lambda.ExpressionParser;
import com.jayantkrish.jklol.ccg.lambda2.Expression2;
import com.jayantkrish.jklol.ccg.lambda2.StaticAnalysis;
import edu.stanford.nlp.sempre.Formula;
import edu.stanford.nlp.sempre.Formulas;
import edu.stanford.nlp.sempre.LambdaFormula;
import fig.basic.LispTree;

public class WikiTablesUtil {
  public static String toPnpLogicalForm(Formula expression) {
    /*
    Sempre's lambda expressions are written differently from what pnp expects. We make the following change
    (lambda x ((reverse fb:cell.cell.number) (var x))) -> (lambda (x) ((reverse fb:cell.cell.number) x))
    We need to do this for all bound variables.
     */
    String expressionString = expression.toString();
    LispTree expressionTree = expression.toLispTree();
    if (expressionTree.isLeaf())
      return expressionString;
    Set<String> boundVariables = new HashSet<>();
    // BFS to find all the free variables
    Queue<LispTree> fringe = new LinkedList<>();
    fringe.add(expressionTree);
    while (!fringe.isEmpty()) {
      LispTree fringeHead = fringe.remove();
      Formula fringeHeadFormula = Formulas.fromLispTree(fringeHead);
      if (fringeHeadFormula instanceof LambdaFormula) {
        boundVariables.add(((LambdaFormula) fringeHeadFormula).var);
      }
      if (!fringeHead.isLeaf()) {
        for (LispTree subTree : fringeHead.children) {
          fringe.add(subTree);
        }
      }
    }
    for (String variable: boundVariables) {
      expressionString = expressionString.replaceAll(String.format("lambda %s", variable),
                                                     String.format("lambda (%s)", variable));
      expressionString = expressionString.replaceAll(String.format("(var %s)", variable), variable);
    }
    return expressionString;
  }

  public static String toSempreLogicalForm(Expression2 expression) {
    Queue<String> variableNames = new LinkedList<>(Arrays.asList("x", "y", "z"));
    for (char varName = 'a'; varName <= 'w'; varName++) {
      variableNames.add(String.valueOf(varName));
    }
    // Find all canonicalized bound variables
    Map<String, String> variableMap = new HashMap<>();
    Queue<Expression2> fringe = new LinkedList<>();
    fringe.add(expression);
    while (!fringe.isEmpty()) {
      Expression2 currExpression = fringe.remove();
      List<Expression2> currChildren = currExpression.getSubexpressions();
      if (currChildren != null) {
        for (Expression2 subExpression : currChildren)
          fringe.add(subExpression);
      }
      if (!StaticAnalysis.isLambda(currExpression))
        continue;
      for (String var : StaticAnalysis.getLambdaArguments(currExpression)) {
        if (var.startsWith("$")) {
          if (!variableMap.containsKey(var))
            variableMap.put(var, variableNames.remove());
        }
      }
    }
    String expressionString = expression.toString();
    for (String var: variableMap.keySet()) {
      String variableName = variableMap.get(var);
      expressionString = expressionString.replaceAll(String.format("lambda \\(\\%s\\)", var),
                                                     String.format("lambda %s", variableName));
      expressionString = expressionString.replaceAll(String.format("\\%s", var),
                                                     String.format("(var %s)", variableName));
      // The last replacement can potentially lead to formulae like ((reverse fb:row.row.player) ((var x))))
      // with a single child in subtree with (var x). Fixing those.
      expressionString = expressionString.replaceAll(String.format("\\(\\(var %s\\)\\)", variableName),
                                                     String.format("(var %s)", variableName));
    }
    return expressionString;
  }
}
