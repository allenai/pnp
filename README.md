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
* **Computation graphs** are a subset of probabilistic neural
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
ln -s <PATH_TO_DYNET>/build/contrib/swig/dynet_swigJNI_scala.jar .
ln -s <PATH_TO_DYNET>/build/contrib/swig/dynet_swigJNI_dylib.jar .
```

That's it! Verify that your installation works by running `sbt test`
in the root directory.

## Usage

This section describes how to use probabilistic neural programs to
define and train a model. The typical usage has three steps:

1. **Define a model.** Models are implemented by writing a function
   that takes your problem input and outputs `Pnp[X]` objects. The
   probabilistic neural program type `Pnp[X]` represents a function
   from neural network parameters to probability distributions over
   values of type `X`. Each program describes a (possibly infinite)
   space of executions, each of which returns a value of type `X`.
2. **Train.** Training is performed by passing a list of examples
   to a `Trainer`, where each example consists of a `Pnp[X]` object
   and a label. Labels are implemented as functions that assign costs
   to program executions or as conditional distributions over correct
   executions. Many training algorithms can be used, from
   loglikelihood to learning-to-search algorithms.
3. **Run the model.** A model can be run on a new input by
   constructing the appropriate `Pnp[X]` object, then running
   inference on this object with trained parameters.

These steps are illustrated in detail for a sequence-to-sequence model
in
[Seq2Seq2.scala](src/main/scala/org/allenai/pnp/examples/Seq2Seq.scala). For
a more complex example, run the [GeoQuery semantic parsing
experiment](experiments/geoquery/scripts/example.sh).

## Defining Probabilistic Neural Programs

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
(which in turn depends on the values of intermediate choices). We 
evaluate `pnp` by running inference, which simultaneously runs the
forward pass of the network and performs probabilistic inference:

```scala
nnParams = ... 
val dist = pnp.beamSearch(10, nnParams)
```

### Choose

The `choose` operator defines a distribution over a list of values:

```scala
val flip: Pnp[Boolean] = choose(Array(true, false), Array(0.5, 0.5))
```

This snippet creates a probability distribution that returns either
true or false with 50% probability. `flip` has type `Pnp[Boolean]`,
which represents a function from neural network parameters to
probability distributions over values of type `Boolean`. (In this case
it's just a probability distribution since we haven't referenced any
parameters.)  Note that `flip` is not a draw from the distribution,
rather, *it is the distribution itself*. The probability of each
choice can be given to `choose` either in an explicit list (as above)
or via an `Expression` of a neural network.

We compose distributions using `for {...} yield {...}`:

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
the value within the for/yield block to construct another probability
distribution. We can now run inference on this object:

```scala
val marginals3 = twoFlips.beamSearch(5)
println(marginals3.marginals().getProbabilityMap)
```

This prints out the expected probabilities:

```
{false=0.75, true=0.25}
```

### Neural Networks

Probabilistic neural programs have access to an underlying computation
graph that is used to define neural networks:

```scala
def mlp(x: FloatVector): Pnp[Boolean] = {
  for {
    // Get the computation graph
    cg <- computationGraph()

    // Get the parameters of a multilayer perceptron by name.
    // The dimensionalities and values of these parameters are 
    // defined in a PnpModel that is passed to inference.
    weights1 <- param("layer1Weights")
    bias1 <- param("layer1Bias")
    weights2 <- param("layer2Weights")

    // Input the feature vector to the computation graph and
    // run the multilayer perceptron to produce scores.
    inputExpression = input(cg.cg, Seq(FEATURE_VECTOR_DIM), x)
    scores = weights2 * tanh((weights1 * inputExpression) + bias1)

     // Choose a label given the scores. Scores is expected to
     // be a 2-element vector, where the first element is the score
     // of true, etc.
     y <- choose(Array(true, false), scores)
  } yield {
    y
  }
}
```

We can then evaluate the network on an example:

```scala
val model = PnpModel.init(true)
// Initialize the network parameters. The values are
// chosen randomly.
model.addParameter("layer1Weights", Seq(HIDDEN_DIM, FEATURE_VECTOR_DIM))
model.addParameter("layer1Bias", Seq(HIDDEN_DIM))
model.addParameter("layer2Weights", Seq(2, HIDDEN_DIM))

// Run the multilayer perceptron on featureVector
val featureVector = new FloatVector(Seq(1.0f, 2.0f, 3.0f))
val dist = mlp(featureVector)
val marginals = dist.beamSearch(2, model)
 
for (x <- marginals.executions) {
  println(x)
}
```

This prints something like:

```
[Execution true -0.4261836111545563]
[Execution false -1.058420181274414]
```

Each execution has a single value that is an output of our program and
a score derived from the neural network computation. In this case, the
scores are log probabilities, but the scores may have different
semantics depending on the way the model is defined and its parameters
are trained.

Pnp uses Dynet as the underlying neural network library, which
provides a rich set of operations (e.g., LSTMs). See the [Dynet
documentation](http://dynet.readthedocs.io/en/latest/operations.html)
for details, along with the documentation for [Dynet Scala
bindings](https://github.com/allenai/dynet/tree/master/swig).

TODO: document usage of RNNBuilders, which have to be used statelessly.

### Defining Richer Models

Probabilistic neural programs can be easily composed to construct
richer models using `for {...} yield {...}`. For example, we can
define a CRF sequence tagger using the multilayer perceptron above:

```scala
def sequenceTag(xs: Seq[FloatVector]): Pnp[List[Boolean]] = {
  xs.foldLeft(Pnp.value(List[Boolean]()))((x, y) => for {
    cur <- mlp(y)
    rest <- x

    cg <- computationGraph()
    _ <- if (rest.length > 0) {
      // Add a factor to the model that scores adjacent labels
      // in the sequence. Here, labelNn runs a neural network
      // whose inputs are cur and the next label, and whose output
      // is a 1-element vector containing the score.
      score(labelNn(cur, rest.head, cg.cg))
    } else {
      value(())
    }
  } yield {
    cur :: rest
  })
}
```

We can now run this model on a sequence of feature vectors in the
same way as the multilayer perceptron:

```scala
// Same model as before, but make it globally normalized 
// and add some more parameters for labelNn
model.locallyNormalized = false
model.addLookupParameter("left", 2, Seq(LABEL_DIM))
model.addLookupParameter("right", 2, Seq(LABEL_DIM))

val featureVectors = Seq(new FloatVector(...), new FloatVector(...), new FloatVector(...))
val dist = sequenceTag(featureVectors)
val marginals = dist.beamSearch(5, model)
for (x <- marginals.executions) {
  println(x)
}
```

This prints something like:

```
[Execution List(true, true, true) 5.28779661655426]
[Execution List(false, true, true) 1.7529568672180176]
[Execution List(true, true, false) 1.4970757961273193]
[Execution List(true, false, false) -0.007531404495239258]
[Execution List(true, false, true) -0.42748916149139404]
```

## Training

TODO

## Inference

TODO
