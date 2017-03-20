#!/bin/bash -e

DATA_DIR="data/dqa_parts_v1"
DIAGRAMS="$DATA_DIR/diagrams.json"
DIAGRAM_FEATURES="$DATA_DIR/diagram_features_xy.json"
DATA_SPLIT="unseen_category"
TRAIN_BEAM="5"
TEST_BEAM="20"
EPOCHS="1"
TRAIN_OPTS=""
TRAIN="$DATA_DIR/data_splits/$DATA_SPLIT/train.json"
TEST="$DATA_DIR/data_splits/$DATA_SPLIT/validation.json"

OUT_DIR="experiments/dqa_parts_v1/output/"
EXPERIMENT_NAME="$DATA_SPLIT/dqa_310/pnp_update/"
EXPERIMENT_DIR="$OUT_DIR/$EXPERIMENT_NAME/"

# MATCHING_MODEL_DIR="$EXPERIMENT_DIR/matching_model.ser"
# INDEPENDENT_MODEL="$EXPERIMENT_DIR/independent_model.ser"
# BINARY_MATCHING_MODEL="$EXPERIMENT_DIR/binary_matching_model.ser"

mkdir -p $EXPERIMENT_DIR
