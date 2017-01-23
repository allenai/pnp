package org.allenai.wikitables;

import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.regex.Pattern;

import com.google.common.collect.ImmutableList;
import com.jayantkrish.jklol.ccg.lambda.Type;
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration;
import com.jayantkrish.jklol.ccg.lambda.AbstractTypeDeclaration;

public class WikiTablesTypeDeclaration extends AbstractTypeDeclaration {

    // Atomic types
    public static final Type NUMBER_TYPE = Type.createAtomic("i");
    // fb:row
    public static final Type ROW_TYPE = Type.createAtomic("r");
    // Cells and parts can be expressed directly with the prefix.
    public static final String CELL_PREFIX = "fb:cell";
    public static final Type CELL_TYPE = Type.createAtomic("c");
    public static final String PART_PREFIX = "fb:part";  
    public static final Type PART_TYPE = Type.createAtomic("p");

    // Functional types
    // Take a row and get its next row
    public static final String ROW_NEXT_STRING = "fb:row.row.next";
    public static final Type ROW_NEXT_FUNCTION = Type.createFunctional(ROW_TYPE, ROW_TYPE, false);

    // Take an index and get a row (at that index)
    public static final String ROW_INDEX_STRING = "fb:row.row.index";
    public static final Type ROW_INDEX_FUNCTION = Type.createFunctional(NUMBER_TYPE, ROW_TYPE, false);

    // Take a number and return a cell
    // Note that we are using the same function type for dates as well.
    public static final String CELL_NUMBER_STRING = "fb:cell.cell.number";
    public static final String CELL_DATE_STRING = "fb:cell.cell.date";
    public static final String CELL_NUM2_STRING = "fb:cell.cell.num2";
    public static final Type CELL_NUMBER_FUNCTION = Type.createFunctional(NUMBER_TYPE, CELL_TYPE, false);

    // Take a string (part of a cell) and return a cell
    public static final String CELL_STR1_STRING = "fb:cell.cell.str1";
    public static final String CELL_STR2_STRING = "fb:cell.cell.str2";
    public static final String CELL_PART_STRING = "fb:cell.cell.part";
    public static final Type CELL_PART_FUNCTION = Type.createFunctional(PART_TYPE, CELL_TYPE, false);

    public static final Map<String, Type> FULLY_SPECIFIED_FUNCTIONS = new HashMap<>();
    static {
        FULLY_SPECIFIED_FUNCTIONS.put(ROW_NEXT_STRING, ROW_NEXT_FUNCTION);
        FULLY_SPECIFIED_FUNCTIONS.put(ROW_INDEX_STRING, ROW_INDEX_FUNCTION);
        FULLY_SPECIFIED_FUNCTIONS.put(CELL_NUMBER_STRING, CELL_NUMBER_FUNCTION);
        FULLY_SPECIFIED_FUNCTIONS.put(CELL_DATE_STRING, CELL_NUMBER_FUNCTION);
        FULLY_SPECIFIED_FUNCTIONS.put(CELL_NUM2_STRING, CELL_NUMBER_FUNCTION);
        FULLY_SPECIFIED_FUNCTIONS.put(CELL_STR1_STRING, CELL_PART_FUNCTION);
        FULLY_SPECIFIED_FUNCTIONS.put(CELL_STR2_STRING, CELL_PART_FUNCTION);
        FULLY_SPECIFIED_FUNCTIONS.put(CELL_PART_STRING, CELL_PART_FUNCTION);
    }
    // Takes a cell and returns a row, indicated by a prefix
    // (Example: fb:row.row.home_town takes a cell in the "home_town" column and returns the row.)
    public static final String ROW_FUNCTION_PREFIX = "fb:row.row";
    public static final Type ROW_CELL_FUNCTION = Type.createFunctional(CELL_TYPE, ROW_TYPE, false);

    public WikiTablesTypeDeclaration(Map<String, String> supertypeMap) {
        super(supertypeMap);
    }

    public WikiTablesTypeDeclaration() {
        this(new HashMap<String, String>());
    }

    @Override
    public Type getType(String constant) {
        if (FULLY_SPECIFIED_FUNCTIONS.containsKey(constant))
            return FULLY_SPECIFIED_FUNCTIONS.get(constant);
        else if (constant.startsWith(CELL_PREFIX))
            return CELL_TYPE;
        else if (constant.startsWith(PART_PREFIX))
            return PART_TYPE;
        else if (constant.startsWith(ROW_FUNCTION_PREFIX))
            return ROW_CELL_FUNCTION;
        else if (constant.matches("\\d*\\.?\\d+"))
            return NUMBER_TYPE;
        else if (constant.equals("number"))
            return Type.createFunctional(NUMBER_TYPE, NUMBER_TYPE, false);
        else if (constant.equalsIgnoreCase("reverse"))
            return Type.parseFrom("<<#2,#1>,<#1,#2>>");
        // Return atomic functions and the one prefix function.
        return TypeDeclaration.TOP;
    }

}
