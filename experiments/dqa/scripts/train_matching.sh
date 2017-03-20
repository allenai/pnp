#!/bin/bash -e

source "experiments/dqa/scripts/config.sh"

echo "Training matching model..."
# ./experiments/dqa/scripts/run.sh org.allenai.dqa.matching.TrainMatchingCli --beamSize $TRAIN_BEAM --epochs $EPOCHS --examples $TRAIN --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --modelOut $MATCHING_MODEL $TRAIN_OPTS > $EXPERIMENT_DIR/matching_train_log.txt

echo "Testing matching model..."
./experiments/dqa/scripts/run.sh org.allenai.dqa.matching.TestMatchingCli --beamSize $TEST_BEAM --examples $TEST --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --model $MATCHING_MODEL --lossJson $EXPERIMENT_DIR/matching_loss.json  > $EXPERIMENT_DIR/matching_test_log_beam=10.txt

mkdir -p $EXPERIMENT_DIR/matching_loss/
python experiments/dqa/scripts/visualize_loss.py $EXPERIMENT_DIR/matching_loss.json $EXPERIMENT_DIR/matching_loss/
tar cf $EXPERIMENT_DIR/matching_loss.tar $EXPERIMENT_DIR/matching_loss/
gzip -f $EXPERIMENT_DIR/matching_loss.tar

# ./experiments/dqa/scripts/run.sh org.allenai.dqa.matching.TestMatchingCli --beamSize $TEST_BEAM --examples $TRAIN --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --model $MATCHING_MODEL --lossJson $EXPERIMENT_DIR/matching_train_loss.json  > $EXPERIMENT_DIR/matching_trainloss_log.txt

echo "Finished training matching"
