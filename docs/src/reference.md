# [Reference](@id reference)

Here we give the definitive specification of the LearnAPI.jl interface. For informal
guides see [Anatomy of an Implementation](@ref) and [Common Implementation
Patterns](@ref).


## [Important terms and concepts](@id scope)

The LearnAPI.jl specification is predicated on a few basic, informally defined notions:


### Data and observations

ML/statistical algorithms are typically applied in conjunction with resampling of
*observations*, as in
[cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)). In this
document *data* will always refer to objects encapsulating an ordered sequence of
individual observations. If an algorithm is trained using multiple data objects, it is
undertood that individual objects share the same number of observations, and that
resampling of one component implies synchronized resampling of the others.

A `DataFrame` instance, from [DataFrames.jl](https://dataframes.juliadata.org/stable/), is
an example of data, the observations being the rows. Typically, data provided to
LearnAPI.jl algorithms, will implement the
[MLUtils.jl](https://juliaml.github.io/MLUtils.jl/stable) `getobs/numobs` interface for
accessing individual observations, but implementations can opt out of this requirement;
see [`obs`](@ref) and [`LearnAPI.data_interface`](@ref) for details.

!!! note

	In the MLUtils.jl
	convention, observations in tables are the rows but observations in a matrix are the
	columns.

### [Hyperparameters](@id hyperparameters)

Besides the data it consumes, a machine learning algorithm's behavior is governed by a
number of user-specified *hyperparameters*, such as the number of trees in a random
forest. In LearnAPI.jl, one is allowed to have hyperparameters that are not data-generic.
For example, a class weight dictionary, which will only make sense for a target taking
values in the set of dictionary keys, can be specified as a hyperparameter.


### [Targets and target proxies](@id proxy)

#### Context

After training, a supervised classifier predicts labels on some input which are then
compared with ground truth labels using some accuracy measure, to assesses the performance
of the classifier. Alternatively, the classifier predicts class probabilities, which are
instead paired with ground truth labels using a proper scoring rule, say. In outlier
detection, "outlier"/"inlier" predictions, or probability-like scores, are similarly
compared with ground truth labels. In clustering, integer labels assigned to observations
by the clustering algorithm can can be paired with human labels using, say, the Rand
index. In survival analysis, predicted survival functions or probability distributions are
compared with censored ground truth survival times. And so on ...

#### Definitions

More generally, whenever we have a variable (e.g., a class label) that can, at least in
principle, be paired with a predicted value, or some predicted "proxy" for that variable
(such as a class probability), then we call the variable a *target* variable, and the
predicted output a *target proxy*. In this definition, it is immaterial whether or not the
target appears in training (is supervised) or whether or not the model generalizes to new
observations ("learns").

LearnAPI.jl provides singleton [target proxy types](@ref proxy_types) for prediction
dispatch. These are also used to distinguish performance metrics provided by the package
[StatisticalMeasures.jl](https://juliaai.github.io/StatisticalMeasures.jl/dev/).


### [Algorithms](@id algorithms)

An object implementing the LearnAPI.jl interface is called an *algorithm*, although it is
more accurately "the configuration of some algorithm".¹ An algorithm encapsulates a
particular set of user-specified [hyperparameters](@ref) as the object's *properties*
(which conceivably differ from its fields). It does not store learned parameters.

Informally, we will sometimes use the word "model" to refer to the output of
`fit(algorithm, ...)` (see below), something which typically does store learned
parameters.

For `algorithm` to be a valid LearnAPI.jl algorithm,
[`LearnAPI.constructor(algorithm)`](@ref) must be defined and return a keyword constructor
enabling recovery of `algorithm` from its properties:

```julia
properties = propertynames(algorithm)
named_properties = NamedTuple{properties}(getproperty.(Ref(algorithm), properties))
@assert algorithm == LearnAPI.constructor(algorithm)(; named_properties...)
```

which can be tested with `@assert `[`LearnAPI.clone(algorithm)`](@ref)` == algorithm`.

Note that if if `algorithm` is an instance of a *mutable* struct, this requirement
generally requires overloading `Base.==` for the struct.

No LearnAPI.jl method is permitted to mutate an algorithm. In particular, one should make
deep copies of RNG hyperparameters before using them in a new implementation of
[`fit`](@ref).

#### Composite algorithms (wrappers)

A *composite algorithm* is one with at least one property that can take other algorithms
as values; for such algorithms [`LearnAPI.is_composite`](@ref)`(algorithm)` must be `true`
(fallback is `false`). Generally, the keyword constructor provided by
[`LearnAPI.constructor`](@ref) must provide default values for all fields that are not
algorithm-valued.

Any object `algorithm` for which [`LearnAPI.functions`](@ref)`(algorithm)` is non-empty is
understood to have a valid implementation of the LearnAPI.jl interface.

#### Example

Any instance of `GradientRidgeRegressor` defined below is a valid algorithm.

```julia
struct GradientRidgeRegressor{T<:Real}
	learning_rate::T
	epochs::Int
	l2_regularization::T
end
GradientRidgeRegressor(; learning_rate=0.01, epochs=10, l2_regularization=0.01) =
	GradientRidgeRegressor(learning_rate, epochs, l2_regularization)
LearnAPI.constructor(::GradientRidgeRegressor) = GradientRidgeRegressor
```

## Documentation

Attach public LearnAPI.jl-related documentation for an algorithm to it's *constructor*,
rather than to the struct defining its type. In this way, an algorithm can implement
multiple interfaces, in addition to the LearnAPI interface, with separate document strings
for each.

## Methods

!!! note "Compulsory methods"

    All new algorithm types must implement [`fit`](@ref),
    [`LearnAPI.algorithm`](@ref algorithm_minimize), [`LearnAPI.constructor`](@ref) and
    [`LearnAPI.functions`](@ref).

Most algorithms will also implement [`predict`](@ref) and/or [`transform`](@ref).

### List of methods

- [`fit`](@ref fit): for training or updating algorithms that generalize to new data. Or,
  for non-generalizing algorithms (see [Static Algorithms](@ref)), for wrapping
  `algorithm` in a mutable struct that can be mutated by `predict`/`transform` to record
  byproducts of those operations.
  
- [`update`](@ref fit): for updating learning outcomes after hyperparameter changes, such
  as increasing an iteration parameter.
  
- [`update_observations`](@ref fit), [`update_features`](@ref fit): update learning
  outcomes by presenting additional training data.

- [`predict`](@ref operations): for outputting [targets](@ref proxy) or [target
  proxies](@ref proxy) (such as probability density functions)

- [`transform`](@ref operations): similar to `predict`, but for arbitrary kinds of output,
  and which can be paired with an `inverse_transform` method

- [`inverse_transform`](@ref operations): for inverting the output of
  `transform` ("inverting" broadly understood)

- [`minimize`](@ref algorithm_minimize): for stripping the `model` output by `fit` of
  inessential content, for purposes of serialization.

- [`LearnAPI.target`](@ref input), [`LearnAPI.weights`](@ref input),
  [`LearnAPI.features`](@ref): for extracting relevant parts of training data, where
  defined.

- [`obs`](@ref data_interface): method for exposing to the user
  algorithm-specific representations of data, which are additionally guaranteed to
  implement the observation access API specified by
  [`LearnAPI.data_interface(algorithm)`](@ref).

- [Accessor functions](@ref accessor_functions): these include functions like
  `feature_importances` and `training_losses`, for extracting, from training outcomes,
  information common to many algorithms.

- [Algorithm traits](@ref traits): methods that promise specific algorithm behavior or
  record general information about the algorithm. Only [`LearnAPI.constructor`](@ref) and
  [`LearnAPI.functions`](@ref) are universally compulsory.


## Utilities

```@docs
LearnAPI.clone
LearnAPI.@trait
```

---

¹ We acknowledge users may not like this terminology, and may know "algorithm" by some
other name, such as "strategy", "options", "hyperparameter set", "configuration", or
"model". Consensus on this point is difficult; see, e.g.,
[this](https://discourse.julialang.org/t/ann-learnapi-jl-proposal-for-a-basement-level-machine-learning-api/93048/20)
Julia Discourse discussion.
