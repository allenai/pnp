package org.allenai.wikitables

import edu.stanford.nlp.sempre.Formula
import org.scalatest.Matchers
import org.scalatest.FlatSpecLike

class WikiTablesEntityLinkerSpec extends FlatSpecLike with Matchers {
  
  import WikiTablesEntityLinker._ 

  def toFormulas(s: Seq[Any]): Seq[Formula] = {
    s.map(x => Formula.fromString(x.toString))
  }

  "tryParseNumber" should "convert ordinal formats" in {
    tryParseNumber("1.0") should be(toFormulas(Seq(1)))
    tryParseNumber("37.0") should be(toFormulas(Seq(37)))
    tryParseNumber("37.2") should be(toFormulas(Seq(37.2)))
    tryParseNumber("null") should be(toFormulas(Seq()))
  }
  
  it should "convert number formats" in {
    tryParseNumber(">4.0") should be(toFormulas(Seq(4)))
    tryParseNumber("<4.0") should be(toFormulas(Seq(4)))
    tryParseNumber(">=1500.0") should be(toFormulas(Seq(1500)))
    tryParseNumber("<=1500.0") should be(toFormulas(Seq(1500)))
    tryParseNumber(">1.5E3") should be(toFormulas(Seq(1500)))
    tryParseNumber("1.0 - 0.0") should be(toFormulas(Seq(1, 0)))
  }
  
  it should "convert percent formats" in {
    tryParseNumber(">%40.0") should be(toFormulas(Seq(40)))
    tryParseNumber("%99.0") should be(toFormulas(Seq(99)))
    tryParseNumber("~%99.0") should be(toFormulas(Seq(99)))
    tryParseNumber("~%99.5") should be(toFormulas(Seq(99.5)))
  }
  
  it should "convert duration formats" in {
    tryParseNumber("P19Y") should be(toFormulas(Seq(19)))
    tryParseNumber("PXY") should be(toFormulas(Seq()))
  }
  
  "tryParseDate" should "convert date formats" in {
    tryParseDate("XXXX-09-03") should be(toFormulas(Seq(0, 9, 3)))
    tryParseDate("2012-02") should be(toFormulas(Seq(2012, 2)))
    tryParseDate("189X") should be(toFormulas(Seq(1890)))
    tryParseNumber("PRESENT_REF") should be(toFormulas(Seq()))
    tryParseNumber("THIS P1000Y") should be(toFormulas(Seq(1000)))
    tryParseNumber("2011/2012") should be(toFormulas(Seq(2011, 2012)))
  }
}