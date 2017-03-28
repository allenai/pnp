#!/bin/bash -e

SCRIPT_DIR="experiments/pascal_parts/scripts/"
DATA_DIR="data/pascal_parts/"
DIAGRAMS="$DATA_DIR/diagrams.json"
DIAGRAM_FEATURES="$DATA_DIR/diagram_features_xy.json"
DATA_SPLIT="unseen_category"
TRAIN_BEAM="5"
TEST_BEAM="20"
EPOCHS="1"
TRAIN_OPTS=""
TRAIN="$DATA_DIR/data_splits_for_ssmn/train.json"
TEST="$DATA_DIR/data_splits_for_ssmn/validation.json"

OUT_DIR="experiments/pascal_parts/output/"
EXPERIMENT_NAME="v1_normalized"
EXPERIMENT_DIR="$OUT_DIR/$EXPERIMENT_NAME/"

mkdir -p $EXPERIMENT_DIR
