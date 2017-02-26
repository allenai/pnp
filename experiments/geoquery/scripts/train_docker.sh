#!/bin/bash -e

# Train a single semantic parsing model on a docker host.
# Run this command with many different configurations to
# sweep parameters, etc.

TRAIN="data/geoquery/all_folds.ccg"
NP_LIST="data/geoquery/np_list.ccg"
TEST="data/geoquery/test.ccg"

EXPERIMENT_NAME="70"
OUT_DIR="experiments/geoquery/output/$EXPERIMENT_NAME/"
LOG=$OUT_DIR/train_log.txt

mkdir -p $OUT_DIR

CLASSPATH=`find lib -name '*.jar' | tr "\\n" :`
java -Djava.library.path=lib -classpath $CLASSPATH org.allenai.pnp.semparse.SemanticParserCli --trainingData $TRAIN --entityData $NP_LIST --testData $TEST > $LOG

