# Probabilistic Neural Programs

Probabilistic Neural Programming (PNP) is a Scala library for
expressing, training and running inference in neural network models
that **include discrete choices**. The enhanced expressivity of PNP is
useful for structured prediction, reinforcement learning, and latent
variable models.

Probabilistic neural programs have several advantages over computation
graph libraries for neural networks, such as TensorFlow:

* **Probabilistic inference** is implemented within the library. For
  example, running a beam search to (approximately) generate the
  highest-scoring output sequence of a sequence-to-sequence model
  takes 1 line of code in PNP.
* **Additional training algorithms** that require running inference
  during training are part of the library. This includes
  learning-to-search algorithms, such as LaSO, reinforcement learning,
  and training latent variable models.
* **Computation graphs** are a strict subset of probabilistic neural
  programs. We use [DyNet](https://github.com/clab/dynet) to express
  neural networks, which provides a rich set of operations and
  efficient training.

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

1. **Define the model.** Models are implemented by writing a function
   that takes your problem input and outputs `Pnp[X]` objects. The
   probabilistic neural program type `Pnp[X]` represents a function
   from neural network parameters to probability distributions over
   values of type `X`. Each program describes a (possibly infinite)
   space of executions, each of which returns a value of type `X`.
2. **Generate labels.** Labels are implemented as functions that assign
   costs to program executions or as conditional distributions over
   correct executions.
3. **Train.** Training is performed by passing a list of examples to a
   `Trainer`, where each example consists of a `Pnp[X]` object and a
   label. Many training algorithms can be used, from loglikelihood to
   learning-to-search algorithms.
4. **Run the model.** A model can be runned on a new input by
   constructing the appropriate `Pnp[X]` object, then running
   inference on this object with trained parameters.

These steps are illustrated in detail for a sequence-to-sequence
neural translation model in
[Seq2Seq2.scala](src/main/scala/org/allenai/pnp/examples/Seq2Seq.scala).

### Defining Probabilistic Neural Programs

Probabilistic neural programs are specified by writing the forward
computation of a neural network, using the `choose` operation to
represent discrete choices. Roughly, we can write: 

```scala
val pnp = for {
  scores1 <- ... some neural net operations ...
  // Make a discrete choice
  x1 <- choose(values, scores1)
  scores2 <- ... more neural net operations, may depend on x1 ...
  ...
  xn <- choose(values, scoresn)
} yield {
  xn
}
```

`pnp` then represents a function that takes some neural network
parameters and returns a distribution over possible values of `xn`
(which in turn depends on the values of intermediate choices). We can
evaluate `pnp` by running inference, which simulatenously runs the
forward pass of the network and performs probabilistic inference.

The `choose` operator defines a distribution over a list of values:

```scala
val flip: Pnp[Boolean] = choose(Seq(true, false), Seq(0.5, 0.5))
```

This snippet creates a probability distribution that returns either
true or false with 50% probability. `flip` has type `Pnp[Boolean]`,
which represents a function from neural network parameters to
probability distributions over values of type `Boolean`. (In this case
it's just a probability distribution since we haven't referenced any
parameters.)  Note that `flip` is not a draw from the distribution,
rather, *it is the distribution itself*. The probability of each
choice can be given to `choose` either in an explicit list (as above)
or via an `Expression` in a neural network.

We can compose distributions using `for {...} yield {...}`:

```scala
val twoFlips: Pnp[Boolean] = for {
  x <- flip
  y <- flip
} yield {
  x && y
}
```

This program returns `true` if two independent draws from `flip` both
return `true`. The notation `x <- flip` can be thought of as drawing a
value from `flip` and assigning it to `x`. However, we can only use
this assignment for the purpose of constructing another probability
distribution.

TODO: finish docs
