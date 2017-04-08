#!/bin/bash -e

SCRIPT_DIR="experiments/wikitables/scripts/"
FOLD=5
# Training data.
# TRAIN="data/WikiTableQuestions/data/subsamples/random-split_1-train_100.examples"
TRAIN="data/WikiTableQuestions/data/random-split-$FOLD-train.examples"
# Development data used for evaluating model accuracy as training progresses.
# TRAIN_DEV="data/WikiTableQuestions/data/subsamples/random-split_1-dev_500.examples"
# TRAIN_DEV="data/WikiTableQuestions/data/subsamples/random-split_1-dev_100.examples"
TRAIN_DEV="data/WikiTableQuestions/data/random-split-$FOLD-dev.examples"
# Development data for evaluating the final trained model.
# DEV="data/WikiTableQuestions/data/subsamples/random-split_1-dev_1000.examples"
DEV="data/WikiTableQuestions/data/random-split-$FOLD-dev.examples"
# DEV="data/WikiTableQuestions/data/subsamples/random-split_1-dev_500.examples"
DERIVATIONS_PATH="data/wikitables/dpd_output/onedir2"
# WORD_EMBEDDINGS="data/wikitables/glove.6B.200d.txt"

EXPERIMENT_NAME="fold$FOLD"
EXPERIMENT_DIR="experiments/wikitables/output/dev_folds_edit_kg/$EXPERIMENT_NAME/"

EPOCHS=40
MAX_TRAINING_DERIVATIONS=100
MAX_TEST_DERIVATIONS=10
BEAM_SIZE=5
TEST_BEAM_SIZE=10
VOCAB=2

# Layer dimensionalities of semantic parser
INPUT_DIM=200
HIDDEN_DIM=100
ACTION_DIM=100
ACTION_HIDDEN_DIM=100

mkdir -p $EXPERIMENT_DIR

# echo $TRAIN
# echo $TRAIN_DEV
# echo $DEV
# echo $EXPERIMENT_NAME
