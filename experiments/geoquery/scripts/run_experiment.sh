#!/bin/bash -e

TRAIN="data/geoquery/all_folds.ccg"
NP_LIST="data/geoquery/np_list.ccg"
TEST="data/geoquery/test.ccg"

EXPERIMENT_NAME="76"
OUT_DIR="experiments/geoquery/output/$EXPERIMENT_NAME/"
MODEL_OUT="experiments/geoquery/output/$EXPERIMENT_NAME/parser.ser"
TRAIN_LOG=$OUT_DIR/train_log.txt
TEST_LOG=$OUT_DIR/test_log.txt

mkdir -p $OUT_DIR

sbt "run-main org.allenai.pnp.semparse.TrainSemanticParserCli --trainingData $TRAIN --entityData $NP_LIST --modelOut $MODEL_OUT" > $TRAIN_LOG
sbt "run-main org.allenai.pnp.semparse.TestSemanticParserCli --testData $TEST --entityData $NP_LIST --model $MODEL_OUT" > $TEST_LOG

