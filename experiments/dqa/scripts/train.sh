
TRAIN="data/dqa/sample/questions.json"
DIAGRAMS="data/dqa/sample/diagrams.json"
DIAGRAM_FEATURES="data/dqa/sample/diagram_features_synthetic.json"

sbt "run-main org.allenai.dqa.labeling.LabelingDqaCli --trainingData $TRAIN --diagrams $DIAGRAMS --diagramFeatures $DIAGRAM_FEATURES" 
