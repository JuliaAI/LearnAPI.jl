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
an example of data, the observations being the rows. LearnAPI.jl makes no assumptions
about how observations can be accessed, except in the case of the output of [`obs`](@ref
data_interface), which must implement the MLUtils.jl `getobs`/`numobs` interface. For
example, it is generally ambiguous whether the rows or columms of a matrix are considered
observations, but if a matrix is returned by [`obs`](@ref data_interface) the observations
must be the columns.

### [Hyperparameters](@id hyperparameters)

Besides the data it consumes, a machine learning algorithm's behavior is governed by a
number of user-specified *hyperparameters*, such as the number of trees in a random
forest. In LearnAPI.jl, one is allowed to have hyperparematers that are not data-generic.
For example, a class weight dictionary will only make sense for a target taking values in
the set of dictionary keys. 


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
compared with censored ground truth survival times.

#### Definitions

More generally, whenever we have a variable (e.g., a class label) that can (in principle)
can be paired with a predicted value, or some predicted "proxy" for that variable (such as
a class probability), then we call the variable a *target* variable, and the predicted
output a *target proxy*. In this definition, it is immaterial whether or not the target
appears in training (is supervised) or whether or not the model generalizes to new
observations ("learns").

LearnAPI.jl provides singleton [target proxy types](@ref proxy_types) for prediction
dispatch in LearnAPI.jl. These are also used to distinguish performance metrics provided
by the package
[StatisticalMeasures.jl](https://juliaai.github.io/StatisticalMeasures.jl/dev/).


### [Algorithms](@id algorithms)

An object implementing the LearnAPI.jl interface is called an *algorithm*, although it is
more accurately "the configuration of some algorithm".¹ It will have a type name
reflecting the name of some ML/statistics algorithm (e.g., `RandomForestRegressor`) and it
will encapsulate a particular set of user-specified [hyperparameters](@ref).

Additionally, for `alg::Alg` to be a LearnAPI algorithm, we require:

- `Base.propertynames(alg)` returns the hyperparameters of `alg`.

- If `alg` is an algorithm, then so are all instances of the same type.

- If `_alg` is another algorithm, then `alg == _alg` if and only if `typeof(alg) ==
  typeof(_alg)` and corresponding properties are `==`. This includes properties that are
  random number generators (which should be copied in training to avoid mutation).

- If an algorithm has other algorithms as hyperparameters, then
  [`LearnAPI.is_composite`](@ref)`(alg)` must be `true` (fallback is `false`).

- A keyword constructor for `Alg` exists, providing default values for *all* non-algorithm
  hyperparameters.
  
- At least one non-trait LearnAPI.jl function must be overloaded for instances of `Alg`,
  and accordingly `LearnAPI.functions(algorithm)` must be non-empty.

Any object `alg` for which [`LearnAPI.functions`](@ref)`(alg)` is non-empty is understood
have a valid implementation of the LearnAPI.jl interface.


### Example

Any instance of `GradientRidgeRegressor` defined below meets all but the last criterion
above:

```julia
struct GradientRidgeRegressor{T<:Real}
	learning_rate::T
	epochs::Int
	l2_regularization::T
end
GradientRidgeRegressor(; learning_rate=0.01, epochs=10, l2_regularization=0.01) =
    GradientRidgeRegressor(learning_rate, epochs, l2_regularization)
```

The same is not true if we make this a `mutable struct`. In that case we will need to
appropriately overload `Base.==` for `GradientRidgeRegressor`.


## Methods

Only these method names are exported: `fit`, `obsfit`, `predict`, `obspredict`,
`transform`, `obstransform`, `inverse_transform`, `minimize`, and `obs`. All new
implementations must implement [`obsfit`](@ref), the accessor function
[`LearnAPI.algorithm`](@ref algorithm_minimize) and the trait
[`LearnAPI.functions`](@ref).

- [`fit`](@ref fit)/[`obsfit`](@ref): for training algorithms that generalize to new data

- [`predict`](@ref operations)/[`obspredict`](@ref): for outputing [targets](@ref proxy)
  or [target proxies](@ref proxy) (such as probability density functions)

- [`transform`](@ref operations)/[`obstransform`](@ref): similar to `predict`, but for
  arbitrary kinds of output, and which can be paired with an `inverse_transform` method

- [`inverse_transform`](@ref operations): for inverting the output of
  `transform` ("inverting" broadly understood)

- [`minimize`](@ref algorithm_minimize): for stripping the `model` output by `fit` of
  inessential content, for purposes of serialization.

- [`obs`](@ref data_interface): a method for exposing to the user "optimized",
  algorithm-specific representations of data, which can be passed to `obsfit`,
  `obspredict` or `obstransform`, but which can also be efficiently resampled using the
  `getobs`/`numobs` interface provided by
  [MLUtils.jl](https://github.com/JuliaML/MLUtils.jl).

- [Accessor functions](@ref accessor_functions): include things like `feature_importances`
  and `training_losses`, for extracting, from training outcomes, information common to
  many algorithms. 

- [Algorithm traits](@ref traits): special methods that promise specific algorithm
  behavior or for recording general information about the algorithm. The only universally
  compulsory trait is `LearnAPI.functions(algorithm)`, which returns a list of the
  explicitly overloaded non-trait methods.
  
---

¹ We acknowledge users may not like this terminology, and may know "algorithm" by some
other name, such as "strategy", "options", "hyperparameter set", "configuration", or
"model". Consensus on this point is difficult; see, e.g.,
[this](https://discourse.julialang.org/t/ann-learnapi-jl-proposal-for-a-basement-level-machine-learning-api/93048/20)
Julia Discourse discussion.
