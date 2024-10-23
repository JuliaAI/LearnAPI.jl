"""
    LearnAPI.target(learner, data) -> target

Return, for each form of `data` supported in a call of the form [`fit(learner,
data)`](@ref), the target variable part of `data`. If `nothing` is returned, the
`learner` does not see a target variable in training (is unsupervised).

# Extended help

## What is a target variable?

Examples of target variables are house prices in realestate pricing estimates, the
"spam"/"not spam" labels in an email spam filtering task, "outlier"/"inlier" labels in
outlier detection, cluster labels in clustering problems, and censored survival times in
survival analysis. For more on targets and target proxies, see the "Reference" section of
the LearnAPI.jl documentation.

## New implementations

A fallback returns `nothing`. Must be implemented if `fit` consumes data including a
target variable.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.target)"; overloaded=true))

"""
target(::Any, data) = nothing

"""
    LearnAPI.weights(learner, data) -> weights

Return, for each form of `data` supported in a call of the form [`fit(learner,
data)`](@ref), the per-observation weights part of `data`. Where `nothing` is returned, no
weights are part of `data`, which is to be interpreted as uniform weighting.

# New implementations

Overloading is optional. A fallback returns `nothing`.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.weights)"; overloaded=true))

"""
weights(::Any, data) = nothing

"""
    LearnAPI.features(learner, data)

Return, for each form of `data` supported in a call of the form [`fit(learner,
data)`](@ref), the "features" part of `data` (as opposed to the target
variable, for example).

The returned object `X` may always be passed to `predict` or `transform`, where
implemented, as in the following sample workflow:

```julia
model = fit(learner, data)
X = features(data)
ŷ = predict(learner, kind_of_proxy, X) # eg, `kind_of_proxy = Point()`
```

The returned object has the same number of observations as `data`. For supervised models
(i.e., where `:(LearnAPI.target) in LearnAPI.functions(learner)`) `ŷ` above is generally
intended to be an approximate proxy for `LearnAPI.target(learner, data)`, the training
target.


# New implementations

That the output can be passed to `predict` and/or `transform`, and has the same number of
observations as `data`, are the only contracts. A fallback returns `first(data)` if `data`
is a tuple, and otherwise returns `data`.

Overloading may be necessary if [`obs(learner, data)`](@ref) is overloaded to return
some learner-specific representation of training `data`. For density estimators, whose
`fit` typically consumes *only* a target variable, you should overload this method to
return `nothing`.

"""
features(learner, data) = _first(data)
_first(data) = data
_first(data::Tuple) = first(data)
# note the factoring above guards against method ambiguities
