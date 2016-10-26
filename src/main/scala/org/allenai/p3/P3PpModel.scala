package org.allenai.p3

import org.allenai.pnp.Pp
import org.allenai.pnp.PpModel

import com.jayantkrish.jklol.ccg.CcgParser
import com.jayantkrish.jklol.ccg.lambda2.Expression2

class P3PpModel(val parser: CcgParser, val pp: PpModel,
    val lfToPp: Expression2 => Pp[AnyRef]) {
}