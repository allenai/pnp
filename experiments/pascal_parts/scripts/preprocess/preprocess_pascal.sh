#!/bin/bash -e

DATA_DIR=data/pascal_parts_matching/pascal_parts_for_matching/images_resize_crop/
SCRIPT_DIR=experiments/dipart/scripts/preprocess/
RAW_ANNOTATIONS=$DATA_DIR/annotation.json
IMAGE_DIR=$DATA_DIR/
# SYNTACTIC_NGRAMS=~/Desktop/syntactic_ngrams/

DIAGRAM_SIZE_OUTPUT=$DATA_DIR/diagram_sizes.txt
OUTPUT=$DATA_DIR/diagrams.json
# NGRAM_OUTPUT=$DATA_DIR/syntactic_ngrams.json
VGG_DIR=data/pascal_parts_matching/images_resize_crop_feat_fc2/
MATCHING_DIR=$DATA_DIR/matchingnet_features/dqa_310/
FEATURE_OUTPUT=$DATA_DIR/diagram_features_xy.json

sips -g pixelHeight -g pixelWidth $IMAGE_DIR/**/*.png > $DIAGRAM_SIZE_OUTPUT
./$SCRIPT_DIR/preprocess_diagram_annotations.py $RAW_ANNOTATIONS $DIAGRAM_SIZE_OUTPUT $OUTPUT
# ./$SCRIPT_DIR/generate_diagram_feats.py $OUTPUT $VGG_DIR $MATCHING_DIR $FEATURE_OUTPUT

