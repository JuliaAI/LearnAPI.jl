# Goals and Approach

## Goals

- Ease of implementation for existing machine learning algorithms

- Applicability to a large variety of algorithms

- Provision of clear interface points for model-generic tooling, such as performance
  evaluation through resampling, hyperparameter optimization, and iterative model control.

- Should be data container agnostic

- Should be documented in detail

It is *not* a design goal of LearnAPI.jl to provide a convenient interface for the general
user to directly interact with ML models. 


## Approach

Machine learning algorithms, also called *models*, have a complicated
taxonomy. Grouping models, or modeling tasks, into a relatively small number of types,
such as "classifier" and "clusterer", and attempting to impose uniform behavior within
each group, is challenging. In our experience developing the [MLJ
ecosystem](https://github.com/alan-turing-institute/MLJ.jl), this either leads to
limitations on the models that can be included in a general interface, or additional
complexity needed to cope with exceptional cases. Even if a complete user interface for
machine learning might benefit from such groupings, a basement-level API for ML should, in
our view, avoid them.

In addition to basic methods, like `fit` and `predict`, LearnAPI provides a number
of optional model
[traits](https://ahsmart.com/pub/holy-traits-design-patterns-and-best-practice-book/),
each promising a specific kind of behavior, such as "The predictions of this model are
probability distributions".  There is no abstract type model hierarchy.

Our preceding remarks notwithstanding, there is, for certain applications involving a
"target" variable (understood in a rather general way - see below) a clear-cut distinction
between models, based on the proxy for the target that is actually output by the
model. Probability distributions, confidence intervals and survival functions are examples
of [Target proxies](@ref). LearnAPI provides a trait for distinguishing such models based
on the target proxy.

LearnAPI is a basement-level interface and not a general ML toolbox. Almost no assumptions
are made about the data manipulated by LearnAPI models. These models can be supervised or
not supervised, can generalize to new data observations, or not generalize.

