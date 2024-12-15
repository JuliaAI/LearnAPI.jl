"""
    LearnAPI.target(learner, observations) -> target

Return, for every conceivable `observations` returned by a call of the form [`obs(learner,
data)`](@ref), the target variable part of `observations`. If `nothing` is returned, the
`learner` does not see a target variable in training (is unsupervised).

The returned object `y` has the same number of observations as `observations` does and is
guaranteed to implement the data interface specified by
[`LearnAPI.data_interface(learner)`](@ref).

# Extended help

## What is a target variable?

Examples of target variables are house prices in real estate pricing estimates, the
"spam"/"not spam" labels in an email spam filtering task, "outlier"/"inlier" labels in
outlier detection, cluster labels in clustering problems, and censored survival times in
survival analysis. For more on targets and target proxies, see the "Reference" section of
the LearnAPI.jl documentation.

## New implementations

A fallback returns `nothing`. The method must be overloaded if [`fit`](@ref) consumes data
that includes a target variable. If `obs` is not being overloaded, then `observations`
above is any `data` supported in calls of the form [`fit(learner, data)`](@ref).  The form
of the output `y` should be suitable for pairing with the output of [`predict`](@ref), in
the evaluation of a loss function, for example.

Ensure the object `y` returned by `LearnAPI.target`, unless `nothing`, implements the data
interface specified by [`LearnAPI.data_interface(learner)`](@ref).

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.target)"; overloaded=true))

"""
target(::Any, observations) = nothing

"""
    LearnAPI.weights(learner, observations) -> weights

Return, for every conceivable `observations` returned by a call of the form [`obs(learner,
data)`](@ref), the weights part of `observations`. Where `nothing` is returned, no weights
are part of `data`, which is to be interpreted as uniform weighting.

The returned object `w` has the same number of observations as `observations` does and is
guaranteed to implement the data interface specified by
[`LearnAPI.data_interface(learner)`](@ref).

# Extended help

# New implementations

Overloading is optional. A fallback returns `nothing`. If `obs` is not being overloaded,
then `observations` above is any `data` supported in calls of the form [`fit(learner,
data)`](@ref).

Ensure the returned object, unless `nothing`, implements the data interface specified by
[`LearnAPI.data_interface(learner)`](@ref).

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.weights)"; overloaded=true))

"""
weights(::Any, observations) = nothing

"""
    LearnAPI.features(learner, observations)

Return, for every conceivable `observations` returned by a call of the form [`obs(learner,
data)`](@ref), the "features" part of `data` (as opposed to the target variable, for
example).

The returned object `X` may always be passed to `predict` or `transform`, where
implemented, as in the following sample workflow:

```julia
observations = obs(learner, data)
model = fit(learner, observations)
X = LearnAPI.features(learner, observations)
ŷ = predict(model, kind_of_proxy, X) # eg, `kind_of_proxy = Point()`
```

For supervised models (i.e., where `:(LearnAPI.target) in LearnAPI.functions(learner)`)
`ŷ` above is generally intended to be an approximate proxy for the target variable.

The object `X` returned by `LearnAPI.features` has the same number of observations as
`observations` does and is guaranteed to implement the data interface specified by
[`LearnAPI.data_interface(learner)`](@ref).

# Extended help

# New implementations

A fallback returns `first(observations)` if `observations` is a tuple, and otherwise
returns `observations`. New implementations may need to overload this method if this
fallback is inadequate.

For density estimators, whose `fit` typically consumes *only* a target variable, you
should overload this method to return `nothing`.  If `obs` is not being overloaded, then
`observations` above is any `data` supported in calls of the form [`fit(learner,
data)`](@ref).

It must otherwise be possible to pass the return value `X` to `predict` and/or
`transform`, and `X` must have same number of observations as `data`.

Ensure the returned object, unless `nothing`, implements the data interface specified by
[`LearnAPI.data_interface(learner)`](@ref).

"""
features(learner, observations) = _first(observations)
_first(observations) = observations
_first(observations::Tuple) = first(observations)
# note the factoring above guards against method ambiguities
