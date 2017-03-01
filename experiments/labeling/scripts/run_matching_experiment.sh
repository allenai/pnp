DIAGRAMS="data/labeling/diagrams.json"
DIAGRAM_FEATURES="data/labeling/diagram_features_synthetic.json"
MODEL="matching_model.ser"

sbt "run-main org.allenai.dqa.matching.TrainMatchingCli --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --modelOut $MODEL"
sbt "run-main org.allenai.dqa.matching.TestMatchingCli --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES --model $MODEL"
