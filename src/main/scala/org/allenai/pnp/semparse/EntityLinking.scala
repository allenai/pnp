package org.allenai.pnp.semparse

import com.jayantkrish.jklol.ccg.lambda.Type

class EntityLinking {
  
  def entities(): List[Entity] = {
    // TODO
    null
  }
}

case class Entity(val id: String, val t: Type, val template: Template, val names: List[List[Int]]) {
}