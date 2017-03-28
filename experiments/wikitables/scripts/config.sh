#!/bin/bash -e

SCRIPT_DIR="experiments/wikitables/scripts/"
TRAIN="data/wikitables/wikitables_sample.examples"
DERIVATIONS_PATH="data/wikitables/dpd_output/onedir2"

EXPERIMENT_NAME="000"
EXPERIMENT_DIR="experiments/wikitables/output/$EXPERIMENT_NAME/"

mkdir -p $EXPERIMENT_DIR

