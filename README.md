# pnp

Scala library for probabilistic neural programming. 

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
