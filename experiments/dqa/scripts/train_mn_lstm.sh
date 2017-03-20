#!/bin/bash -e

source "experiments/dqa/scripts/config.sh"

MY_NAME=matching_lstm
MY_DIR=$EXPERIMENT_DIR/$MY_NAME/
MY_MODEL=$MY_DIR/model.ser
MY_FLAGS="--lstmEncode --matchIndependent --loglikelihood"
MY_EPOCHS=5

mkdir -p $MY_DIR

echo "Training $MY_NAME model..."
./experiments/dqa/scripts/run.sh org.allenai.dqa.matching.TrainMatchingCli --beamSize $TRAIN_BEAM --epochs $MY_EPOCHS --examples $TRAIN --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --modelOut $MY_MODEL $TRAIN_OPTS $MY_FLAGS > $MY_DIR/log.txt

echo "Testing $MY_NAME model..."
./experiments/dqa/scripts/run.sh org.allenai.dqa.matching.TestMatchingCli --beamSize $TEST_BEAM --examples $TEST --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --model $MY_MODEL --lossJson $MY_DIR/validation_error_independent.json  > $MY_DIR/validation_error_independent_log.txt

./experiments/dqa/scripts/run.sh org.allenai.dqa.matching.TestMatchingCli --beamSize 120 --enforceMatching --globalNormalize --examples $TEST --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --model $MY_MODEL --lossJson $MY_DIR/validation_error_matching.json  > $MY_DIR/validation_error_matching_log.txt

# mkdir -p $MY_DIR/validation_error/
# python experiments/dqa/scripts/visualize_loss.py $MY_DIR/validation_error.json $MY_DIR/validation_error/
# tar cf $MY_DIR/validation_error.tar $MY_DIR/validation_error/
# gzip -f $MY_DIR/validation_error.tar

# ./experiments/dqa/scripts/run.sh org.allenai.dqa.matching.TestMatchingCli --beamSize $TEST_BEAM --examples $TRAIN --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --model $MY_MODEL --lossJson $MY_DIR/train_error.json  > $MY_DIR/train_error_log.txt

# mkdir -p $MY_DIR/train_error/
# python experiments/dqa/scripts/visualize_loss.py $MY_DIR/train_error.json $MY_DIR/train_error/
# tar cf $MY_DIR/train_error.tar $MY_DIR/train_error/
# gzip -f $MY_DIR/train_error.tar

echo "Finished training $MY_NAME"

