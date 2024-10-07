"""
    LearnAPI.target(algorithm, data) -> target

Return, for each form of `data` supported in a call of the form [`fit(algorithm,
data)`](@ref), the target variable part of `data`. If `nothing` is returned, the
`algorithm` does not see a target variable in training (is unsupervised).

Refer to LearnAPI.jl documentation for the precise meaning of "target".

# New implementations

A fallback returns `nothing`. Must be implemented if `fit` consumes data including a
target variable.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.target)"; overloaded=true))

"""
target(::Any, data) = nothing

"""
    LearnAPI.weights(algorithm, data) -> weights

Return, for each form of `data` supported in a call of the form [`fit(algorithm,
data)`](@ref), the per-observation weights part of `data`. Where `nothing` is returned, no
weights are part of `data`, which is to be interpreted as uniform weighting.

# New implementations

Overloading is optional. A fallback returns `nothing`.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.weights)"; overloaded=true))

"""
weights(::Any, data) = nothing

"""
    LearnAPI.features(algorithm, data)

Return, for each form of `data` supported in a call of the form [`fit(algorithm,
data)`](@ref), the "features" part of `data` (as opposed to the target
variable, for example).

The returned object `X` may always be passed to `predict` or `transform`, where
implemented, as in the following sample workflow:

```julia
model = fit(algorithm, data)
X = features(data)
ŷ = predict(algorithm, kind_of_proxy, X) # eg, `kind_of_proxy = Point()`
```

The returned object has the same number of observations as `data`. For supervised models
(i.e., where `:(LearnAPI.target) in LearnAPI.functions(algorithm)`) `ŷ` above is generally
intended to be an approximate proxy for `LearnAPI.target(algorithm, data)`, the training
target.


# New implementations

That the output can be passed to `predict` and/or `transform`, and has the same number of
observations as `data`, are the only contracts. A fallback returns `first(data)` if `data`
is a tuple, and otherwise returns `data`.

Overloading may be necessary if [`obs(algorithm, data)`](@ref) is overloaded to return
some algorithm-specific representation of training `data`. For density estimators, whose
`fit` typically consumes *only* a target variable, you should overload this method to
return `nothing`.

"""
features(algorithm, data) = _first(data)
_first(data) = data
_first(data::Tuple) = first(data)
# note the factoring above guards agains method ambiguities
