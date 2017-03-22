#!/bin/bash -e

TQA_DIR=data/tqa/
TQA=$TQA_DIR/tqa_dataset_beta_v8.json
IMAGE_DIR=$TQA_DIR/
TQA_DIAGRAMS=$TQA_DIR/tqa_diagrams.json

cat $TQA | jq -c '.[] | .diagramAnnotations | to_entries | .[]' > $TQA_DIAGRAMS

sips -g pixelHeight -g pixelWidth $IMAGE_DIR/**/*.png > $DIAGRAM_SIZE_OUTPUT
