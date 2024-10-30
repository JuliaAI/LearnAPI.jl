"""
    LearnAPI.target(learner, data) -> target

Return, for each form of `data` supported in a call of the form [`fit(learner,
data)`](@ref), the target variable part of `data`. If `nothing` is returned, the
`learner` does not see a target variable in training (is unsupervised).

The returned object `y` has the same number of observations as `data`. If `data` is the
output of an [`obs`](@ref) call, then `y` is additionally guaranteed to implement the
data interface specified by [`LearnAPI.data_interface(learner)`](@ref).

# Extended help

## What is a target variable?

Examples of target variables are house prices in real estate pricing estimates, the
"spam"/"not spam" labels in an email spam filtering task, "outlier"/"inlier" labels in
outlier detection, cluster labels in clustering problems, and censored survival times in
survival analysis. For more on targets and target proxies, see the "Reference" section of
the LearnAPI.jl documentation.

## New implementations

A fallback returns `nothing`. The method must be overloaded if `fit` consumes data
including a target variable.

If overloading [`obs`](@ref), ensure that the return value, unless `nothing`, implements
the data interface specified by [`LearnAPI.data_interface(learner)`](@ref), in the special
case that `data` is the output of an `obs` call.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.target)"; overloaded=true))

"""
target(::Any, data) = nothing

"""
    LearnAPI.weights(learner, data) -> weights

Return, for each form of `data` supported in a call of the form [`fit(learner,
data)`](@ref), the per-observation weights part of `data`. Where `nothing` is returned, no
weights are part of `data`, which is to be interpreted as uniform weighting.

The returned object `w` has the same number of observations as `data`. If `data` is the
output of an [`obs`](@ref) call, then `w` is additionally guaranteed to implement the
data interface specified by [`LearnAPI.data_interface(learner)`](@ref).

# Extended help

# New implementations

Overloading is optional. A fallback returns `nothing`.

If overloading [`obs`](@ref), ensure that the return value, unless `nothing`, implements
the data interface specified by [`LearnAPI.data_interface(learner)`](@ref), in the special
case that `data` is the output of an `obs` call.

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
X = LearnAPI.features(learner, data)
ŷ = predict(learner, kind_of_proxy, X) # eg, `kind_of_proxy = Point()`
```

For supervised models (i.e., where `:(LearnAPI.target) in LearnAPI.functions(learner)`)
`ŷ` above is generally intended to be an approximate proxy for `LearnAPI.target(learner,
data)`, the training target.

The object `X` returned by `LearnAPI.target` has the same number of observations as
`data`. If `data` is the output of an [`obs`](@ref) call, then `X` is additionally
guaranteed to implement the data interface specified by
[`LearnAPI.data_interface(learner)`](@ref).

# Extended help

# New implementations

For density estimators, whose `fit` typically consumes *only* a target variable, you
should overload this method to return `nothing`.

It must otherwise be possible to pass the return value `X` to `predict` and/or
`transform`, and `X` must have same number of observations as `data`. A fallback returns
`first(data)` if `data` is a tuple, and otherwise returns `data`.

Further overloadings may be necessary to handle the case that `data` is the output of
[`obs(learner, data)`](@ref), if `obs` is being overloaded. In this case, be sure that
`X`, unless `nothing`, implements the data interface specified by
[`LearnAPI.data_interface(learner)`](@ref).

"""
features(learner, data) = _first(data)
_first(data) = data
_first(data::Tuple) = first(data)
# note the factoring above guards against method ambiguities
