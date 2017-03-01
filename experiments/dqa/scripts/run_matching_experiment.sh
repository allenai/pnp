#!/bin/bash -e

DIAGRAMS="data/dqa/diagrams.json"
DIAGRAM_FEATURES="data/dqa/diagram_features_synthetic.json"
MODEL="matching_model.ser"

sbt "run-main org.allenai.dqa.matching.TrainMatchingCli --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --modelOut $MODEL"
sbt "run-main org.allenai.dqa.matching.TestMatchingCli --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --model $MODEL"
