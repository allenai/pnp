#!/bin/bash -e

DATA_DIR=data/dqa/
RAW_ANNOTATIONS=$DATA_DIR/all_output.json
IMAGE_DIR=$DATA_DIR/images/

DIAGRAM_SIZE_OUTPUT=$DATA_DIR/diagram_sizes.txt
OUTPUT=$DATA_DIR/diagrams.json
FEATURE_OUTPUT=data/dqa/diagram_features_synthetic.json

sips -g pixelHeight -g pixelWidth $IMAGE_DIR/**/*.png > $DIAGRAM_SIZE_OUTPUT
python experiments/dqa/scripts/preprocess_diagram_annotations.py $RAW_ANNOTATIONS $DIAGRAM_SIZE_OUTPUT $OUTPUT
python experiments/dqa/scripts/generate_diagram_feats.py $OUTPUT $FEATURE_OUTPUT
