#!/bin/bash -e

DIAGRAMS="data/dqa/diagrams.json"
DIAGRAM_FEATURES="data/dqa/diagram_features_synthetic.json"

OUT_DIR="experiments/dqa/output/"
EXPERIMENT_NAME="distances_learned"
EXPERIMENT_DIR="$OUT_DIR/$EXPERIMENT_NAME/"

MATCHING_MODEL="$EXPERIMENT_DIR/matching_model.ser"
INDEPENDENT_MODEL="$EXPERIMENT_DIR/independent_model.ser"
BINARY_MATCHING_MODEL="$EXPERIMENT_DIR/binary_matching_model.ser"

mkdir -p $EXPERIMENT_DIR

echo "Training binary_matching model..."
# sbt "run-main org.allenai.dqa.matching.TrainMatchingCli --binaryFactors --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --modelOut $BINARY_MATCHING_MODEL" > $EXPERIMENT_DIR/binary_matching_train_log.txt

echo "Testing binary_matching model..."
# sbt "run-main org.allenai.dqa.matching.TestMatchingCli --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --model $BINARY_MATCHING_MODEL --lossJson $EXPERIMENT_DIR/binary_matching_loss.json"  > $EXPERIMENT_DIR/binary_matching_test_log.txt

python experiments/dqa/scripts/visualize_loss.py $EXPERIMENT_DIR/binary_matching_loss.json $EXPERIMENT_DIR/binary_matching_loss.html

echo "Training matching model..."
# sbt "run-main org.allenai.dqa.matching.TrainMatchingCli --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --modelOut $MATCHING_MODEL" > $EXPERIMENT_DIR/matching_train_log.txt

echo "Testing matching model..."
# sbt "run-main org.allenai.dqa.matching.TestMatchingCli --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --model $MATCHING_MODEL --lossJson $EXPERIMENT_DIR/matching_loss.json"  > $EXPERIMENT_DIR/matching_test_log.txt

echo "Training independent model..."
# sbt "run-main org.allenai.dqa.matching.TrainMatchingCli --matchIndependent --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --modelOut $INDEPENDENT_MODEL" > $EXPERIMENT_DIR/independent_train_log.txt

echo "Testing independent model..."
# sbt "run-main org.allenai.dqa.matching.TestMatchingCli --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --model $INDEPENDENT_MODEL --lossJson $EXPERIMENT_DIR/independent_loss.json"  > $EXPERIMENT_DIR/independent_test_log.txt

