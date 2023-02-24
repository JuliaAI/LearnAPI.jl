# Goals and Approach

## Goals

- Ease of implementation for existing ML/statistics algorithms

- Breadth of applicability

- Flexibility in extending functionality

- Provision of clear interface points for algorithm-generic tooling, such as performance
  evaluation through resampling, hyperparameter optimization, and iterative algorithm
  control.

- Should make minimal assumptions about data containers

- Should be documented in detail

In particular, the first three goals are to take precedence over user convenience, which
is addressed with a separate, [User Interface](@ref).


## Approach

ML/Statistics algorithms have a complicated taxonomy. Grouping algorithms, or modelling
tasks, into a relatively small number of categories, such as "classification" and
"clusterering", and then imposing uniform behavior within each group, is challenging. In
our experience developing the [MLJ
ecosystem](https://github.com/alan-turing-institute/MLJ.jl), this either leads to
limitations on the algorithms that can be included in a general interface, or additional
complexity needed to cope with exceptional cases. Even if a complete data science
framework might benefit from such groupings, a basement-level API should, in our view,
avoid them.

In addition to basic methods, like `fit` and `predict`, LearnAPI provides a number of
optional algorithm
[traits](https://ahsmart.com/pub/holy-traits-design-patterns-and-best-practice-book/),
each promising a specific kind of behavior, such as "This algorithm supports class
weights".  There is no abstract type hierarchy for ML/statistics algorithms.

LearnAPI.jl intentionally focuses on the notion of [target variables and target
proxies](@ref proxy), which can exist in both the superised and unsupervised setting,
rather than on the supervised/unsupervised dichotomy. In this view a supervised model is
simply one which has a target variable *and* whose target variable appears in training.

LearnAPI is a basement-level interface and not a general ML/statistics toolbox. Algorithms
can be supervised or not supervised, can generalize to new data observations (i.e.,
"learn") or not generalize (e.g., "one-shot" clusterers).

