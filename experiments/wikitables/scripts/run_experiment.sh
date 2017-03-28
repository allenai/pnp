#!/bin/bash -e

source "experiments/wikitables/scripts/config.sh"

MY_NAME="parser"
MY_DIR=$EXPERIMENT_DIR/$MY_NAME/
MY_MODEL=$MY_DIR/model.ser

mkdir -p $MY_DIR

echo "Training $MY_NAME model..."
./$SCRIPT_DIR/run.sh org.allenai.wikitables.WikiTablesSemanticParserCli -trainingData $TRAIN  -derivationsPath $DERIVATIONS_PATH --modelOut $MY_MODEL --skipActionSpaceValidation &> $MY_DIR/train_log.txt 

echo "Evaluating $MY_NAME training error..."
./$SCRIPT_DIR/run.sh org.allenai.wikitables.TestWikiTablesCli -testData $TRAIN  -derivationsPath $DERIVATIONS_PATH --model $MY_MODEL &> $MY_DIR/train_error_log.txt
