#!/bin/bash -e

SCRIPT_DIR="experiments/wikitables/scripts/"
# TRAIN="data/WikiTableQuestions/data/subsamples/random-split_1-train_1000.examples"
# DEV="data/WikiTableQuestions/data/subsamples/random-split_1-dev_1000.examples"
TRAIN="data/WikiTableQuestions/data/random-split-1-train.examples"
DEV="data/WikiTableQuestions/data/random-split-1-dev.examples"
DERIVATIONS_PATH="data/wikitables/dpd_output/onedir2"

EXPERIMENT_NAME="all_001"
EXPERIMENT_DIR="experiments/wikitables/output/$EXPERIMENT_NAME/"

EPOCHS=50
MAX_TRAINING_DERIVATIONS=1
MAX_TEST_DERIVATIONS=10
BEAM_SIZE=5

mkdir -p $EXPERIMENT_DIR
