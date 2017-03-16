#!/bin/bash -e

DATA_DIR="data/dqa_parts_v1"
DIAGRAMS="$DATA_DIR/diagrams.json"
DIAGRAM_FEATURES="$DATA_DIR/diagram_features_xy.json"
DATA_SPLIT="unseen_category"
TRAIN_BEAM="5"
TEST_BEAM="5"
EPOCHS="1"
TRAIN="$DATA_DIR/data_splits/$DATA_SPLIT/train.json"
TEST="$DATA_DIR/data_splits/$DATA_SPLIT/validation.json"

OUT_DIR="experiments/dqa_parts_v1/output/"
EXPERIMENT_NAME="$DATA_SPLIT/dqa_210/partname_features_bias/"
EXPERIMENT_DIR="$OUT_DIR/$EXPERIMENT_NAME/"

MATCHING_MODEL="$EXPERIMENT_DIR/matching_model.ser"
INDEPENDENT_MODEL="$EXPERIMENT_DIR/independent_model.ser"
BINARY_MATCHING_MODEL="$EXPERIMENT_DIR/binary_matching_model.ser"

mkdir -p $EXPERIMENT_DIR

# echo "Training binary_matching model..."
# sbt "run-main org.allenai.dqa.matching.TrainMatchingCli --beamSize $TRAIN_BEAM --epochs $EPOCHS --binaryFactors --examples $TRAIN --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --modelOut $BINARY_MATCHING_MODEL" > $EXPERIMENT_DIR/binary_matching_train_log.txt
# 
# echo "Testing binary_matching model..."
# sbt "run-main org.allenai.dqa.matching.TestMatchingCli --beamSize $TEST_BEAM --examples $TEST --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --model $BINARY_MATCHING_MODEL --lossJson $EXPERIMENT_DIR/binary_matching_loss.json"  > $EXPERIMENT_DIR/binary_matching_test_log.txt
# 
# mkdir -p $EXPERIMENT_DIR/binary_matching_loss/
# python experiments/dqa/scripts/visualize_loss.py $EXPERIMENT_DIR/binary_matching_loss.json $EXPERIMENT_DIR/binary_matching_loss/
# tar cf $EXPERIMENT_DIR/binary_matching_loss.tar $EXPERIMENT_DIR/binary_matching_loss/
# gzip -f $EXPERIMENT_DIR/binary_matching_loss.tar
# 
# # sbt "run-main org.allenai.dqa.matching.TestMatchingCli --beamSize $TEST_BEAM --examples $TRAIN --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --model $BINARY_MATCHING_MODEL --lossJson $EXPERIMENT_DIR/binary_matching_train_loss.json"  > $EXPERIMENT_DIR/binary_matching_trainloss_log.txt

echo "Training matching model..."
sbt "run-main org.allenai.dqa.matching.TrainMatchingCli --beamSize $TRAIN_BEAM --epochs $EPOCHS --examples $TRAIN --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --modelOut $MATCHING_MODEL" > $EXPERIMENT_DIR/matching_train_log.txt

echo "Testing matching model..."
sbt "run-main org.allenai.dqa.matching.TestMatchingCli --beamSize $TEST_BEAM --examples $TEST --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --model $MATCHING_MODEL --lossJson $EXPERIMENT_DIR/matching_loss.json"  > $EXPERIMENT_DIR/matching_test_log.txt

mkdir -p $EXPERIMENT_DIR/matching_loss/
python experiments/dqa/scripts/visualize_loss.py $EXPERIMENT_DIR/matching_loss.json $EXPERIMENT_DIR/matching_loss/
tar cf $EXPERIMENT_DIR/matching_loss.tar $EXPERIMENT_DIR/matching_loss/
gzip -f $EXPERIMENT_DIR/matching_loss.tar

# sbt "run-main org.allenai.dqa.matching.TestMatchingCli --beamSize $TEST_BEAM --examples $TRAIN --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --model $MATCHING_MODEL --lossJson $EXPERIMENT_DIR/matching_train_loss.json"  > $EXPERIMENT_DIR/matching_trainloss_log.txt

echo "Training independent model..."
sbt "run-main org.allenai.dqa.matching.TrainMatchingCli --beamSize $TRAIN_BEAM --epochs $EPOCHS --matchIndependent --examples $TRAIN --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --modelOut $INDEPENDENT_MODEL" > $EXPERIMENT_DIR/independent_train_log.txt

echo "Testing independent model..."
sbt "run-main org.allenai.dqa.matching.TestMatchingCli --beamSize $TEST_BEAM --examples $TEST --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --model $INDEPENDENT_MODEL --lossJson $EXPERIMENT_DIR/independent_loss.json"  > $EXPERIMENT_DIR/independent_test_log.txt

mkdir -p $EXPERIMENT_DIR/independent_loss/
python experiments/dqa/scripts/visualize_loss.py $EXPERIMENT_DIR/independent_loss.json $EXPERIMENT_DIR/independent_loss/
tar cf $EXPERIMENT_DIR/independent_loss.tar $EXPERIMENT_DIR/independent_loss/
gzip -f $EXPERIMENT_DIR/independent_loss.tar

# sbt "run-main org.allenai.dqa.matching.TestMatchingCli --beamSize $TEST_BEAM --examples $TRAIN --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --model $INDEPENDENT_MODEL --lossJson $EXPERIMENT_DIR/independent_train_loss.json"  > $EXPERIMENT_DIR/independent_trainloss_log.txt
