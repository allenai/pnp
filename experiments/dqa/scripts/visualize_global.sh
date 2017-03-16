#!/bin/bash -e

source "experiments/dqa/scripts/config.sh"

EXPERIMENT_NAME="$DATA_SPLIT/dqa_310/final1_laso/"
EXPERIMENT_DIR="$OUT_DIR/$EXPERIMENT_NAME/"
BINARY_MATCHING_MODEL="$EXPERIMENT_DIR/binary_matching_model.ser"

SOURCE="antelope/antelope_0003.png"
TARGET="antelope/antelope_0000.png"
LABELS_TO_MATCH="horn,belly,tail,leg"
SOURCE_QUERY="neck"

./experiments/dqa/scripts/run.sh org.allenai.dqa.matching.VisualizeMatchingCli --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --model $BINARY_MATCHING_MODEL --source $SOURCE --target $TARGET --labelsToMatch $LABELS_TO_MATCH --sourcePart $SOURCE_QUERY --numGrid 25

