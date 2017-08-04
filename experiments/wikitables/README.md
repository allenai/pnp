# Neural Semantic Parsing with Type Constraints for Semi-Structured Tables

This directory contains code and directions for training a neural semantic parser on the
[WikiTableQuestions data set](https://nlp.stanford.edu/software/sempre/wikitable/).
The parser is described in the following paper:

**Neural Semantic Parsing with Type Constraints for Semi-Structured Tables**  
EMNLP 2017  
Jayant Krishnamurthy, Pradeep Dasigi, Matt Gardner  
[PDF](http://ai2-website.s3.amazonaws.com/publications/wikitables.pdf)

## Getting Started

The parser is implemented as a Probabilistic Neural Program (PNP).
Follow these [instructions](../../README.md#installation) to install DyNet and verify that PNP is working.

Next, download the data. From the root directory, run:

```
./experiments/wikitables/scripts/download_data.sh
```

This command will download the v1.0.2 release of WikiTableQuestions (from [here](https://github.com/ppasupat/WikiTableQuestions/releases)) to `data/WikiTableQuestions`.
It will also download the output of dynamic programming on denotations on this data set to `data/dpd_output`.

Finally, run an experiment with the parser:

```
# sbt assembly may require lots of RAM.
export JAVA_OPTS="-Xmx4g"
# Build a fat jar containing the parser code and all of its dependencies.
# The first time you run this command, downloading dependencies may take a while.
sbt assembly
# Train and evaluate the parser.
./experiments/wikitables/scripts/run_experiment.sh
```

This command trains the parser using the configuration stored in `experiments/wikitables/scripts/config.sh`.
The default config file trains and evaluates on a small subsample of the data set (TODO: how long is training time).
The output of the experiment is stored under `experiments/wikitables/output/00/fold1/parser/`, where you will find several files:

* `train_log.txt` -- the output of the training script. Run `less -RS train_log.txt`, then type `F` to watch training as it progresses.
* `parser_final.ser` -- the trained semantic parsing model
* `dev_error_log.txt` -- predictions and evaluation results of the trained parser on the dev set. These results may differ slightly from the official evaluation results as they use different answer matching code.
* `official_results.txt` -- official dev set evaluation results calculated using the `evaluator.py` script included with WikiTableQuestions.

To train on the full data set, edit the file paths in config.
See the file for instructions.
The config file also specifies some model and training parameters.
Other model parameters can be changed by editing the command-line flags in `run_experiment.sh`.

## Scala Code

The scala code for the parser is located
[here](../../src/main/scala/org/allenai/wikitables/).  There are two
command-line programs in this directory. `WikiTablesSemanticParserCli`
trains the parser, and `TestWikiTablesCli` tests the parser.  The
scripts above run both of these programs in sequence to perform an
experiment.
