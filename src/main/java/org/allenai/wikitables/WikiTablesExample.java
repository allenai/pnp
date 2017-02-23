package org.allenai.wikitables;

import java.util.Set;
import java.util.List;
import java.util.ArrayList;

import com.jayantkrish.jklol.ccg.DependencyStructure; 
import com.jayantkrish.jklol.ccg.CcgExample;
import com.jayantkrish.jklol.ccg.lambda2.Expression2; 
import com.jayantkrish.jklol.nlpannotation.AnnotatedSentence;

/**
 * Similar to CcgExample, except that this supports multiple logcal forms per example.
 * This correctly does not hold a syntactic parse or dependencies. But we may need them in the future.
 */

public class WikiTablesExample extends CcgExample {

  public final Set<Expression2> logicalForms;
  public WikiTablesExample(AnnotatedSentence sentence, Set<Expression2> logicalForms) {
    // We are making the logicalForm in CcgExample null, and defining the set logicalForms in this class.
    super(sentence, null, null, null);
    this.logicalForms = logicalForms;
  }

  public Set<Expression2> getLogicalForms() {
    return logicalForms;
  }

  @Override
  public boolean hasLogicalForm() {
    return logicalForms != null;
  }

  @Override
  public String toString() {
    String superString = super.toString();
    List<String> logicalFormStrings = new ArrayList<>();
    for (Expression2 lf : logicalForms) {
        logicalFormStrings.add(lf.toString());
    }
    return superString + " [\n" + String.join(" ", logicalFormStrings) + "\n]";
  }
}
