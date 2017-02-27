#!/bin/bash -e

TRAIN="data/geoquery/all_folds.ccg"
NP_LIST="data/geoquery/np_list.ccg"
TEST="data/geoquery/test.ccg"

MODEL_OUT="parser.ser"

sbt "run-main org.allenai.pnp.semparse.TrainSemanticParserCli --trainingData $TRAIN --entityData $NP_LIST --modelOut $MODEL_OUT" 
sbt "run-main org.allenai.pnp.semparse.TestSemanticParserCli --testData $TEST --entityData $NP_LIST --model $MODEL_OUT" 

