#!/bin/bash -e

source "experiments/wikitables/scripts/config.sh"

MY_NAME="parser"
MY_DIR=$EXPERIMENT_DIR/$MY_NAME/
MY_MODEL=$MY_DIR/parser_final.ser
MODEL_DIR=$MY_DIR/models/

mkdir -p $MY_DIR
mkdir -p $MODEL_DIR

echo "Training $MY_NAME model..."
./$SCRIPT_DIR/run.sh org.allenai.wikitables.WikiTablesSemanticParserCli --trainingData $TRAIN --devData $TRAIN_DEV --derivationsPath $DERIVATIONS_PATH --modelOut $MY_MODEL --modelDir $MODEL_DIR --epochs $EPOCHS --beamSize $BEAM_SIZE --maxDerivations $MAX_TRAINING_DERIVATIONS --vocabThreshold $VOCAB --inputDim $INPUT_DIM --hiddenDim $HIDDEN_DIM --actionDim $ACTION_DIM --actionHiddenDim $ACTION_HIDDEN_DIM --skipActionSpaceValidation &> $MY_DIR/train_log.txt 

echo "Evaluating $MY_NAME training error..."
# ./$SCRIPT_DIR/run.sh org.allenai.wikitables.TestWikiTablesCli --testData $TRAIN --model $MY_MODEL --beamSize $BEAM_SIZE --derivationsPath $DERIVATIONS_PATH --maxDerivations $MAX_TEST_DERIVATIONS &> $MY_DIR/train_error_log.txt

echo "Evaluating $MY_NAME development error..."
./$SCRIPT_DIR/run.sh org.allenai.wikitables.TestWikiTablesCli --testData $DEV --model $MY_MODEL --beamSize $BEAM_SIZE --derivationsPath $DERIVATIONS_PATH --maxDerivations $MAX_TEST_DERIVATIONS &> $MY_DIR/dev_error_log.txt
