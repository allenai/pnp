# Structured Set Matching Networks for One-Shot Part Labeling

This directory contains scripts for running the experiments from
"Structured Set Matching Networks for One-Shot Part Labeling." (TODO:
arXiv link) The corresponding Scala code is located in
`org.allenai.dqa.matching` package.

## Data Set

TODO

## Running Experiments

Once the data is downloaded and preprocessed, you can train and
evaluate the SSMN model by running the following from the root `pnp`
directory:

```
sbt assembly
./experiments/dipart/scripts/train_ssmn.sh
```

This script sends its output to the
`experiments/dipart/output/.../ssmn` directory. After the script
completes, this directory will contain several files:

* `log.txt` shows the progress of model training and corresponding statistics.
* `model.ser` is the serialized trained model.
* `validation_error_log.txt` is the validation error results. The end
  of this file contains the accuracy numbers reported in the paper.
* `validation_error.json` is a JSON representation of the model's validation set predictions.

It will also automatically create an HTML visualization of the model's
predictions in the `validation_error` subdirectory. To view it, simply
open `index.html` in a web browser. The visualization includes
per-category error rates, confusion matrices, and the predicted part
labeling for each example.

The `./experiments/dipart/scripts/` directory contains several other
scripts for training the baselines from the paper. These scripts
similarly send their output to `experiments/dipart/output/`. In some
cases the files may be slightly different than those above. For
example, the matching network (`train_mn_lstm.sh`) has two validation
error logs, one that enforces the matching constraint at test time and
one that doesn't.
