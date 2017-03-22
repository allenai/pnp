#!/bin/bash -e

source "experiments/dipart/scripts/config.sh"

MY_NAME=ssmn_unary
MY_DIR=$EXPERIMENT_DIR/$MY_NAME/
MY_MODEL=$MY_DIR/model.ser
MY_FLAGS="--matchingNetwork --partClassifier"

mkdir -p $MY_DIR

echo "Training $MY_NAME model..."
./$SCRIPT_DIR/run.sh org.allenai.dqa.matching.TrainMatchingCli --beamSize $TRAIN_BEAM --epochs $EPOCHS --examples $TRAIN --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --modelOut $MY_MODEL $TRAIN_OPTS $MY_FLAGS > $MY_DIR/log.txt

echo "Testing $MY_NAME model..."
./$SCRIPT_DIR/run.sh org.allenai.dqa.matching.TestMatchingCli --beamSize $TEST_BEAM --examples $TEST --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --model $MY_MODEL --lossJson $MY_DIR/validation_error.json  > $MY_DIR/validation_error_log.txt

mkdir -p $MY_DIR/validation_error/
python $SCRIPT_DIR/visualize/visualize_loss.py $MY_DIR/validation_error.json $MY_DIR/validation_error/
tar cf $MY_DIR/validation_error.tar $MY_DIR/validation_error/
gzip -f $MY_DIR/validation_error.tar

# ./$SCRIPT_DIR/run.sh org.allenai.dqa.matching.TestMatchingCli --beamSize $TEST_BEAM --examples $TRAIN --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --model $MY_MODEL --lossJson $MY_DIR/train_error.json  > $MY_DIR/train_error_log.txt

# mkdir -p $MY_DIR/train_error/
# python $SCRIPT_DIR/visualize/visualize_loss.py $MY_DIR/train_error.json $MY_DIR/train_error/
# tar cf $MY_DIR/train_error.tar $MY_DIR/train_error/
# gzip -f $MY_DIR/train_error.tar

echo "Finished training $MY_NAME"

