#!/bin/bash -e

TRAIN="data/geoquery/all_folds.ccg"
NP_LIST="data/geoquery/np_list.ccg"
TEST="data/geoquery/test.ccg"

EXPERIMENT_NAME="71"
OUT_DIR="experiments/geoquery/output/$EXPERIMENT_NAME/"
MODEL_OUT="experiments/geoquery/output/$EXPERIMENT_NAME/parser.ser"
LOG=$OUT_DIR/train_log.txt

mkdir -p $OUT_DIR

sbt "run-main org.allenai.pnp.semparse.SemanticParserCli --trainingData $TRAIN --entityData $NP_LIST --testData $TEST --modelOut $MODEL_OUT" > $LOG

