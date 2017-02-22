# Probabilistic Neural Programs

Probabilistic Neural Programming (PNP) is a Scala library for
expressing, training and running inference in neural network models
that *include discrete choices*. The enhanced expressivity of PNP
useful for structured prediction, reinforcement learning, and latent
variable models.

Probabilistic neural programs have several advantages over computation
graph libraries for neural networks, such as TensorFlow:

* Probabilistic inference is part of the library. For example, running
  a beam search to generate the highest-scoring output sequence of a
  sequence-to-sequence model takes 1 line of code in PNP.
* Training algorithms that require running inference during training
  are part of the library. This includes learning-to-search
  algorithms, such as LaSO, reinforcement learning, and training
  latent variable models.
* Computation graphs are a subset of probabilistic neural programs. We
  use [DyNet](https://github.com/clab/dynet) to express neural
  networks, which provides a rich set of operations and efficient
  training.

## Installation

This library depends on DyNet with the
[Scala DyNet bindings](https://github.com/allenai/dynet/tree/master/swig).
See the link for build instructions. After building this library, run
the following commands from the `pnp` root directory:

```
cd lib
ln -s <PATH_TO_DYNET>/build/swig/dynet_swigJNI_scala.jar .
ln -s <PATH_TO_DYNET>/build/swig/libdynet_swig.jnilib .
```

That's it! Verify that your installation works by running `sbt test`
in the root directory.

## Usage

This section describes how to use probabilistic neural programs to
define and train a model. The typical usage has four steps:

1. **Define the model** Models are implemented by writing a function
   that takes your problem input and outputs `Pnp[X]` objects. The
   probabilistic neural program type `Pnp[X]` represents a function
   from neural network parameters to probability distributions over
   values of type `X`. Each program describes a (possibly infinite)
   space of executions, each of which returns some value of type `X`.
2. **Generate labels** Labels are implemented as functions that assign
   costs to program executions or as conditional distributions over
   correct executions.
3. **Train** Training is performed by passing a list of examples to a
   `Trainer`, where each example consists of a `Pnp[X]` object and a
   label. Many training algorithms can be used, from loglikelihood to
   learning-to-search algorithms.
4. **Run the model** A model can be runned on a new input by
   constructing the appropriate `Pnp[X]` object, then running
   inference on this object with trained parameters.

These steps are illustrated in detail for a sequence-to-sequence
neural translation model in
[Seq2Seq2.scala](tree/master/src/main/scala/org/allenai/pnp/examples/Seq2Seq.scala).

TODO: finish docs
First, import:

```scala

```




