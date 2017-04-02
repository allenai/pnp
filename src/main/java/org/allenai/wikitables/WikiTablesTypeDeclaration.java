package org.allenai.wikitables;

import java.util.*;

import com.google.common.collect.Maps;
import com.jayantkrish.jklol.ccg.lambda.AbstractTypeDeclaration;
import com.jayantkrish.jklol.ccg.lambda.Type;
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration;

public class WikiTablesTypeDeclaration extends AbstractTypeDeclaration {

  // Atomic types
  public static final Type NUMBER_TYPE = Type.createAtomic("i");
  public static final Type DATE_TYPE = Type.createAtomic("d");
  public static final Type ROW_TYPE = Type.createAtomic("r");
  public static final Type ENTITY_TYPE = Type.createAtomic("e");
  public static final Type CELL_TYPE = Type.createAtomic("c");
  public static final Type PART_TYPE = Type.createAtomic("p");
  public static final Type VAR_TYPE = Type.createAtomic("v");
  
  public static final Type COL_FUNCTION_TYPE = Type.parseFrom("<c,r>");

  private static final String[][] FUNCTION_TYPES = {
    // Atomic types
    {"fb:type.row", "r"},
    {"fb:type.cell", "c"},
    {"fb:type.part", "p"},

    // Take a row and get the next row
    {"fb:row.row.next", "<r,r>"},
    // Take an index and get the row at that index
    {"fb:row.row.index", "<i,r>"},

    // Interpret a cell as a number or date (?)
    {"fb:cell.cell.number", "<i,c>"},
    {"fb:cell.cell.num2", "<i,c>"},
    {"fb:cell.cell.date", "<d,c>"},

    // Get parts of a cell.
    {"fb:cell.cell.str1", "<p,c>"},
    {"fb:cell.cell.str2", "<p,c>"},
    {"fb:cell.cell.part", "<p,c>"},

    // Maps a type of object to all entities with that type.
    // To get this to typecheck, we've treated types as elements of
    // that type.
    {"fb:type.object.type", "<#1,#1>"},

    {"number", "<i,i>"},
    {"date", "<i,<i,<i,d>>>"},
    //{"var", "<#1,#1>"},

    // Quantifiers.
    // TODO: only dates and numbers (i.e., comparable things) can be bound to #2
    {"argmax", "<i,<i,<#1,<<#2,#1>,#1>>>>"},
    {"argmin", "<i,<i,<#1,<<#2,#1>,#1>>>>"},

    // TODO: only dates and numbers can be bound to #1
    {"max", "<#1,#1>"},
    {"min", "<#1,#1>"},

    {"count", "<#1,i>"},
    {"reverse", "<<#2,#1>,<#1,#2>>"},

    {"or", "<#1,<#1,#1>>"},
    {"and", "<#1,<#1,#1>>"},

    {"!=", "<#1,#1>"},
    {">=", "<#1,#1>"},
    {"<=", "<#1,#1>"},
    {">", "<#1,#1>"},
    {"<", "<#1,#1>"},

    {"avg", "<i,i>"},
    {"sum", "<i,i>"},

    {"-", "<i,<i,i>>"},
  };

  private static final Map<String, Type> FUNCTION_TYPE_MAP = Maps.newHashMap();
  static {
    for (int i = 0; i < FUNCTION_TYPES.length; i++) {
      FUNCTION_TYPE_MAP.put(FUNCTION_TYPES[i][0], Type.parseFrom(FUNCTION_TYPES[i][1]));
    }
  }

  // Takes a cell and returns a row, indicated by a prefix
  // (Example: fb:row.row.home_town takes a cell in the "home_town" column and returns the row.)
  public static final String CELL_PREFIX = "fb:cell";
  public static final String PART_PREFIX = "fb:part";
  public static final String ROW_FUNCTION_PREFIX = "fb:row.row";
  public static final Type ROW_CELL_FUNCTION = Type.createFunctional(CELL_TYPE, ROW_TYPE, false);

  public static final Map<String, String> SUPERTYPE_MAP = new HashMap<>();
  static {
    SUPERTYPE_MAP.put("r", "e");
    SUPERTYPE_MAP.put("c", "e");
    SUPERTYPE_MAP.put("p", "e");
    SUPERTYPE_MAP.put("d", "i");
  }

  public WikiTablesTypeDeclaration() {
    super(SUPERTYPE_MAP);
  }

  @Override
  public Type getType(String constant) {
    if (FUNCTION_TYPE_MAP.containsKey(constant))
      return FUNCTION_TYPE_MAP.get(constant);
    else if (constant.startsWith(CELL_PREFIX))
      return CELL_TYPE;
    else if (constant.startsWith(PART_PREFIX))
      return PART_TYPE;
    else if (constant.startsWith(ROW_FUNCTION_PREFIX))
      return ROW_CELL_FUNCTION;
    else if (constant.matches("[-+]?\\d*\\.?\\d+"))
      return NUMBER_TYPE;
    else { 
      return TypeDeclaration.BOTTOM;
    }
  }
}
