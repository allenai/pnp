
TRAIN="data/labeling/questions.json"
DIAGRAMS="data/labeling/diagrams.json"

sbt "run-main org.allenai.dqa.labeling.LabelingDqaCli --trainingData $TRAIN --diagrams $DIAGRAMS" 
