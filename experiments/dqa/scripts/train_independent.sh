#!/bin/bash -e

source "experiments/dqa/scripts/config.sh"

echo "Training independent model..."
./experiments/dqa/scripts/run.sh org.allenai.dqa.matching.TrainMatchingCli --beamSize $TRAIN_BEAM --epochs $EPOCHS --matchIndependent --examples $TRAIN --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --modelOut $INDEPENDENT_MODEL $TRAIN_OPTS > $EXPERIMENT_DIR/independent_train_log.txt

echo "Testing independent model..."
./experiments/dqa/scripts/run.sh org.allenai.dqa.matching.TestMatchingCli --beamSize $TEST_BEAM --examples $TEST --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --model $INDEPENDENT_MODEL --lossJson $EXPERIMENT_DIR/independent_loss.json  > $EXPERIMENT_DIR/independent_test_log.txt

mkdir -p $EXPERIMENT_DIR/independent_loss/
python experiments/dqa/scripts/visualize_loss.py $EXPERIMENT_DIR/independent_loss.json $EXPERIMENT_DIR/independent_loss/
tar cf $EXPERIMENT_DIR/independent_loss.tar $EXPERIMENT_DIR/independent_loss/
gzip -f $EXPERIMENT_DIR/independent_loss.tar

# ./experiments/dqa/scripts/run.sh org.allenai.dqa.matching.TestMatchingCli --beamSize $TEST_BEAM --examples $TRAIN --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --model $INDEPENDENT_MODEL --lossJson $EXPERIMENT_DIR/independent_train_loss.json  > $EXPERIMENT_DIR/independent_trainloss_log.txt

echo "Finished training independent"
