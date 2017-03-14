#!/bin/bash -e

DATA_DIR=data/dqa_parts_v1/
RAW_ANNOTATIONS=$DATA_DIR/annotation.json
IMAGE_DIR=$DATA_DIR/
SYNTACTIC_NGRAMS=~/Desktop/syntactic_ngrams/

DIAGRAM_SIZE_OUTPUT=$DATA_DIR/diagram_sizes.txt
OUTPUT=$DATA_DIR/diagrams.json
NGRAM_OUTPUT=$DATA_DIR/syntactic_ngrams.json
VGG_DIR=$DATA_DIR/vgg_features/dqa_matching_final_complete_working_crop_feat_fc2/
# MATCHING_DIR=$DATA_DIR/matchingnet_features/dqa_37/
# MATCHING_DIR=$DATA_DIR/matchingnet_features/dqa_seg_12/
# Current best:
# MATCHING_DIR=$DATA_DIR/matchingnet_features/dqa_51/
# MATCHING_DIR=$DATA_DIR/matchingnet_features/dqa_61/
# Ani's features
# MATCHING_DIR=$DATA_DIR/matchingnet_features/inference_uc_gray400/
# MATCHING_DIR=$DATA_DIR/matchingnet_features/inference_uc_dist400_pool4_mm6_epoch21/
# MATCHING_DIR=$DATA_DIR/matchingnet_features/inference_uc_dist400_pool4_mm6/
# MATCHING_DIR=$DATA_DIR/matchingnet_features/inference_uc_dist400_pool4_mm6_epoch49_location/
# MATCHING_DIR=$DATA_DIR/matchingnet_features/dqa_61_no_context/
MATCHING_DIR=$DATA_DIR/matchingnet_features/dqa_61_full_context_100/
# MATCHING_DIR=$DATA_DIR/matchingnet_features/dqa_61_point_aware_context_100/
FEATURE_OUTPUT=$DATA_DIR/diagram_features_xy.json

UNSEEN_SAMPLE=$DATA_DIR/unseen_sample_trvats.json
UNSEEN_S_DIR=$DATA_DIR/data_splits/unseen_sample
UNSEEN_SAMPLE_TRAIN=$UNSEEN_S_DIR/train.json
UNSEEN_SAMPLE_VAL=$UNSEEN_S_DIR/validation.json
UNSEEN_SAMPLE_TEST=$UNSEEN_S_DIR/test.json

UNSEEN_CATEGORY=$DATA_DIR/unseen_category_trvats.json
UNSEEN_C_DIR=$DATA_DIR/data_splits/unseen_category
UNSEEN_CATEGORY_TRAIN=$UNSEEN_C_DIR/train.json
UNSEEN_CATEGORY_VAL=$UNSEEN_C_DIR/validation.json
UNSEEN_CATEGORY_TEST=$UNSEEN_C_DIR/test.json

sips -g pixelHeight -g pixelWidth $IMAGE_DIR/**/*.png > $DIAGRAM_SIZE_OUTPUT
./experiments/dqa/scripts/preprocess_diagram_annotations.py $RAW_ANNOTATIONS $DIAGRAM_SIZE_OUTPUT $OUTPUT
./experiments/dqa/scripts/generate_diagram_feats.py $OUTPUT $VGG_DIR $MATCHING_DIR $FEATURE_OUTPUT

# This takes a long time and doesn't need to be recomputed
# python ./experiments/dqa/scripts/preprocess_ngrams.py $OUTPUT $SYNTACTIC_NGRAMS $NGRAM_OUTPUT

# Generate data splits. Note that the sampling is seeded so as to be repeatable
# (as long as the number of samples doesn't change.)
./experiments/dqa/scripts/sample_pairs.py $UNSEEN_SAMPLE $UNSEEN_SAMPLE_TRAIN train -1 -1
./experiments/dqa/scripts/sample_pairs.py $UNSEEN_SAMPLE $UNSEEN_SAMPLE_VAL val -1 -1
./experiments/dqa/scripts/sample_pairs.py $UNSEEN_SAMPLE $UNSEEN_SAMPLE_TEST test -1 -1
./experiments/dqa/scripts/sample_pairs.py $UNSEEN_CATEGORY $UNSEEN_CATEGORY_TRAIN train -1 -1
./experiments/dqa/scripts/sample_pairs.py $UNSEEN_CATEGORY $UNSEEN_CATEGORY_VAL val -1 -1
./experiments/dqa/scripts/sample_pairs.py $UNSEEN_CATEGORY $UNSEEN_CATEGORY_TEST test -1 -1

# Unseen sample splits for different numbers of training diagrams
SPLITS=( 2 5 10 20 )
for i in ${SPLITS[@]}; do
    DIR=$DATA_DIR/data_splits/unseen_sample_$i
    mkdir -p $DIR
    TRAIN=$DATA_DIR/data_splits/unseen_sample_$i/train.json
    VAL=$DATA_DIR/data_splits/unseen_sample_$i/validation.json
    TEST=$DATA_DIR/data_splits/unseen_sample_$i/test.json

    python experiments/dqa/scripts/sample_pairs.py $UNSEEN_SAMPLE $TRAIN train 1 $i
    python experiments/dqa/scripts/sample_pairs.py $UNSEEN_SAMPLE $VAL val -1 -1
    python experiments/dqa/scripts/sample_pairs.py $UNSEEN_SAMPLE $TEST test -1 -1
done
