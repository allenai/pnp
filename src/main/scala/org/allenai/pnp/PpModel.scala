package org.allenai.pnp

import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.dynet._
import scala.collection.mutable.MapBuilder
import scala.collection.mutable.ListBuffer

/** A neural probabilistic program model consisting
  * of a collection of named Tensor parameters. These
  * parameters are used to initialize the computation
  * graph of a program during inference.
  */
class PpModel(var names: Map[String, Parameter], var lookupNames: Map[String, LookupParameter], 
    val model: Model, val locallyNormalized: Boolean) {

  def addParameter(name: String, dim: Dim): Parameter = {
    val param = model.add_parameters(dim)
    names += (name -> param)
    param
  }
  
  def addLookupParameter(name: String, lookupNum: Long, dim: Dim): LookupParameter = {
    val param = model.add_lookup_parameters(lookupNum, dim)
    lookupNames += (name -> param)
    param
  }

  def getParameter(name: String): Parameter = {
    names(name)
  }

  def getLookupParameter(name: String): LookupParameter = {
    lookupNames(name)
  }

  def getComputationGraph(cg: ComputationGraph): CompGraph = {
    new CompGraph(cg, model, names, lookupNames, locallyNormalized)
  }

  def save(saver: ModelSaver): Unit = {
    saver.add_model(model)
    saver.add_boolean(locallyNormalized)
    
    saver.add_int(names.size)
    for ((k, v) <- names) {
      saver.add_object(k)
      saver.add_parameter(v)
    }

    saver.add_int(lookupNames.size)
    for ((k, v) <- lookupNames) {
      saver.add_object(k)
      saver.add_lookup_parameter(v)
    }
  }
}

object PpModel {
  def init(locallyNormalized: Boolean): PpModel = {
    new PpModel(Map(), Map(), new Model, locallyNormalized)
  }
  
  def load(loader: ModelLoader): PpModel = {
    val model = loader.load_model()
    val locallyNormalized = loader.load_boolean()
    
    val numParams = loader.load_int()
    val params = ListBuffer[(String, Parameter)]()
    for (i <- 0 until numParams) {
      val name = loader.load_object(classOf[String])
      val param = loader.load_parameter()
      params += ((name, param))
    }

    val numLookups = loader.load_int()
    val lookups = ListBuffer[(String, LookupParameter)]()
    for (i <- 0 until numLookups) {
      val name = loader.load_object(classOf[String])
      val param = loader.load_lookup_parameter()
      lookups += ((name, param))
    }

    new PpModel(params.toMap, lookups.toMap, model, locallyNormalized)
  }
}