#!/bin/bash -e

source "experiments/wikitables/scripts/config.sh"

MY_NAME="parser"
MY_DIR=$EXPERIMENT_DIR/$MY_NAME/
MODEL_DIR=$MY_DIR/models/
MY_MODEL=$MODEL_DIR/parser_16.ser
TEST_BEAM=10

mkdir -p $MY_DIR
mkdir -p $MODEL_DIR

echo "Evaluating $MY_NAME development error..."
./$SCRIPT_DIR/run.sh org.allenai.wikitables.TestWikiTablesCli --testData $DEV --model $MY_MODEL --beamSize $TEST_BEAM --maxDerivations $MAX_TEST_DERIVATIONS &> $MY_DIR/dev_error_16_beam_10_log.txt
