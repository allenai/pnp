#!/bin/bash -e

# Download the WikiTableQuestions data set.
OUT_DIR=data/
cd $OUT_DIR
wget https://github.com/ppasupat/WikiTableQuestions/releases/download/v1.0.2/WikiTableQuestions-1.0.2-compact.zip
unzip WikiTableQuestions-1.0.2-compact.zip
cd WikiTableQuestions/data
mkdir -p subsamples
head -n100 random-split-1-train.examples > subsamples/random-split-1-train-100.examples
head -n100 random-split-1-dev.examples > subsamples/random-split-1-dev-100.examples
cd ../..

# Download DPD output
wget http://cs.stanford.edu/~ppasupat/research/h-strict-all-matching-lfs.tar.gz
tar xvzf h-strict-all-matching-lfs.tar.gz
# Move all of the output files to a single directory.
mkdir -p dpd_output
cd h-strict-all-matching-lfs
for d in $( ls ); do
    path="$d/*"
    cp $path ../dpd_output/
done

# Download caseless corenlp models
cd ../..
cd lib
wget http://nlp.stanford.edu/software/stanford-corenlp-caseless-2015-04-20-models.jar
