package org.allenai.dqa.labeling

object LabelingUtil {
  
  val UNK = "<UNK>"

  def tokenize(language: String): Array[String] = {
    // The first set of characters are always mapped to their own
    // token. The second set gets a token containing any non-space
    // characters to the right.
    language.toLowerCase().replaceAll("([:&,?./\\(\\)-])", " $1 ")
      .replaceAll("(['])", " $1").split("[ ]+")
  }
}