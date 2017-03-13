package org.allenai.pnp.semparse

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.{Map => MutableMap}

import org.allenai.pnp.Env

import com.jayantkrish.jklol.ccg.CcgExample
import com.jayantkrish.jklol.ccg.lambda.Type
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration
import com.jayantkrish.jklol.ccg.lambda2.StaticAnalysis
import com.jayantkrish.jklol.training.NullLogFunction
import com.jayantkrish.jklol.util.CountAccumulator

import edu.cmu.dynet._

object SemanticParserUtils {
  
  val DYNET_PARAMS = Map(
    "dynet-mem" -> "1024"
  )

  /**
   * Count the number of occurrences of each word type
   * in a collection of examples. 
   */
  def getWordCounts(examples: Seq[CcgExample]): CountAccumulator[String] = {
    val acc = CountAccumulator.create[String]
    for (ex <- examples) {
      ex.getSentence.getWords.asScala.map(x => acc.increment(x, 1.0)) 
    }
    acc
  }
  
  /**
   * Checks that the logical forms in {@code examples}
   * are well-typed using {@code typeDeclaration}. 
   */
  def validateTypes(examples: Seq[CcgExample], typeDeclaration: TypeDeclaration): Unit = {
    for (e <- examples) {
      val lf = e.getLogicalForm
      val typeInference = StaticAnalysis.typeInference(lf, TypeDeclaration.TOP, typeDeclaration)
      
      val constraints = typeInference.getSolvedConstraints
      val typeMap = typeInference.getExpressionTypes.asScala
      
      if (!constraints.isSolvable) {
        println(lf)
        println(typeInference.getConstraints)
        println(typeInference.getSolvedConstraints)
        
        for (i <- 0 until lf.size()) {
          if (typeMap.contains(i)) {
            val t = typeMap(i)
            println("    " + i + " " + t + " " + lf.getSubexpression(i))
          }
        }
      } else {
        for (i <- 0 until lf.size()) {
          if (typeMap.contains(i)) {
            val t = typeMap(i)
            if (isBadType(t)) {
              println(lf)
              println("  B " + i + " " + t + " " + lf.getSubexpression(i))
            }
          }
        }
      }
    }
  }

  private def isBadType(t: Type): Boolean = {
    if (t.isAtomic) {
      if (t.hasTypeVariables || t.equals(TypeDeclaration.TOP) || t.equals(TypeDeclaration.BOTTOM)) {
        true
      } else {
        false
      }
    } else {
      return isBadType(t.getArgumentType) || isBadType(t.getReturnType)
    }
  }

  /** Verify that the parser can generate the logical form
   * in each training example when the search is constrained
   * by the execution oracle.  
   */
  def validateActionSpace(examples: Seq[CcgExample], parser: SemanticParser,
      typeDeclaration: TypeDeclaration): Unit = {
    println("")
    var maxParts = 0
    var numFailed = 0
    val usedRules = ListBuffer[(Type, Template)]()
    for (e <- examples) {
      val sent = e.getSentence
      val tokenIds = sent.getAnnotation("tokenIds").asInstanceOf[Array[Int]]
      val entityLinking = sent.getAnnotation("entityLinking").asInstanceOf[EntityLinking]

      val oracleOpt = parser.generateExecutionOracle(e.getLogicalForm, entityLinking, typeDeclaration)
      val dist = parser.parse(tokenIds, entityLinking)

      if (oracleOpt.isDefined) {
        val oracle = oracleOpt.get
        ComputationGraph.renew()
        val results = dist.beamSearch(1, 50, Env.init, oracle,
            parser.model.getComputationGraph(), new NullLogFunction())
        if (results.executions.size != 1) {
          println("ERROR: " + e + " " + results)
          println("  " + e.getSentence.getWords)
          println("  " + e.getLogicalForm)
          println("  " + e.getSentence.getAnnotation("entityLinking"))

          numFailed += 1
        } else {
          val numParts = results.executions(0).value.parts.size
          maxParts = Math.max(numParts, maxParts)
          if (results.executions.length > 1) {
            println("MULTIPLE: " + results.executions.length + " " + e)
            println("  " + e.getSentence.getWords)
            println("  " + e.getLogicalForm)
            println("  " + e.getSentence.getAnnotation("entityLinking"))
          } else {
            // println("OK   : " + numParts + " " + " "
          }
        }
        
        // Accumulate the rules used in each example
        usedRules ++= oracle.holeTypes.zip(oracle.templates)

        // Print out the rules used to generate this logical form.
        /*
        println(e.getLogicalForm)
        for (t <- oracle.templates) {
          println("  " + t)
        }
        */
        
      } else {
        println("ORACLE: " + e)
        println("  " + e.getSentence.getWords)
        println("  " + e.getLogicalForm)
        println("  " + e.getSentence.getAnnotation("entityLinking"))

        numFailed += 1
      }
    }
    println("max parts: " + maxParts)
    println("decoding failures: " + numFailed)
    
    val holeTypes = usedRules.map(_._1).toSet
    val countMap = MutableMap[Type, CountAccumulator[Template]]()
    for (t <- holeTypes) {
      countMap(t) = CountAccumulator.create()
    }
    
    for ((t, template) <- usedRules) {
      countMap(t).increment(template, 1.0)
    }
    
    for (t <- holeTypes) {
      println(t)
      val counts = countMap(t)
      for (template <- counts.getSortedKeys.asScala) {
        val count = counts.getCount(template)
        println("  " + count + " " + template) 
      }
    }
  }
}