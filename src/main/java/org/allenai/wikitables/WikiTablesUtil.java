package org.allenai.wikitables;

import java.util.*;

import com.jayantkrish.jklol.ccg.lambda2.Expression2;
import com.jayantkrish.jklol.ccg.lambda2.ExpressionSimplifier;
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
      if (!StaticAnalysis.isLambda(currExpression))
        continue;
      for (String var : StaticAnalysis.getLambdaArguments(currExpression)) {
        if (var.startsWith("$")) {
          if (!variableMap.containsKey(var))
            variableMap.put(var, variableNames.remove());
        }
      }
      for (Expression2 subExpression: currExpression.getSubexpressions())
        fringe.add(subExpression);
    }
    String expressionString = expression.toString();
    for (String var: variableMap.keySet()) {
      String variableName = variableMap.get(var);
      expressionString = expressionString.replaceAll(String.format("lambda \\(\\%s\\)", var),
                                                     String.format("lambda %s", variableName));
      expressionString = expressionString.replaceAll(String.format("\\%s", var),
                                                     String.format("(var %s)", variableName));
    }
    return expressionString;
  }

  public static void main(String[] args) {
    Expression2 subExpression = Expression2.nested(Expression2.constant("+"),
            Expression2.constant("x"),
            Expression2.constant("y"));
    Expression2 pnpExpression = Expression2.lambda(Arrays.asList("x"), subExpression);
    pnpExpression = Expression2.lambda(Arrays.asList("y"), pnpExpression);
    ExpressionSimplifier simplifier = ExpressionSimplifier.lambdaCalculus();
    System.out.println("Before simplification:" + pnpExpression.toString());
    pnpExpression = simplifier.apply(pnpExpression);
    System.out.println("After simplification:" + pnpExpression.toString());
    System.out.println("After conversion:" + WikiTablesUtil.toSempreLogicalForm(pnpExpression));
  }
}
