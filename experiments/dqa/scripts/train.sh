
TRAIN="data/dqa/questions.json"
DIAGRAMS="data/dqa/diagrams.json"

sbt "run-main org.allenai.dqa.labeling.LabelingDqaCli --trainingData $TRAIN --diagrams $DIAGRAMS" 
