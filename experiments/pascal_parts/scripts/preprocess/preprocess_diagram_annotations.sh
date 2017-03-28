#!/bin/bash -e

DATA_DIR=data/pascal_parts/
SCRIPT_DIR=experiments/pascal_parts/scripts/preprocess/
RAW_ANNOTATIONS=$DATA_DIR/pascal_parts_for_matching/images/annotation_normalized.json
# IMAGE_DIR=$DATA_DIR/pascal_parts_for_matching/images/

# DIAGRAM_SIZE_OUTPUT=$DATA_DIR/diagram_sizes.txt
OUTPUT=$DATA_DIR/diagrams.json
MATCHING_DIR=$DATA_DIR/pascal_parts_22/
FEATURE_OUTPUT=$DATA_DIR/diagram_features_xy.json

# This command seems to work from the command line but not in the script ??
# echo $IMAGE_DIR/**/*.png | xargs sips -g pixelHeight -g pixelWidth > $DIAGRAM_SIZE_OUTPUT
./$SCRIPT_DIR/preprocess_diagram_annotations.py $RAW_ANNOTATIONS $OUTPUT
./$SCRIPT_DIR/generate_diagram_feats.py $OUTPUT $MATCHING_DIR $FEATURE_OUTPUT

