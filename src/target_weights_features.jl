"""
    LearnAPI.target(algorithm, data) -> target

Return, for each form of `data` supported in a call of the form [`fit(algorithm,
data)`](@ref), the target variable part of `data`. If `nothing` is returned, the
`algorithm` does not see a target variable in training (is unsupervised).

Refer to LearnAPI.jl documenation for the precise meaning of "target".

# New implementations

A fallback returns `nothing`. Must be implemented if `fit` consumes data including a
target variable.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.target)"; overloaded=true))

"""
target(::Any, data) = nothing

"""
    LearnAPI.weights(algorithm, data) -> weights

Return, for each form of `data` supported in a call of the form `[`fit(algorithm,
data)`](@ref), the per-observation weights part of `data`. Where `nothing` is returned, no
weights are part of `data`, which is to be interpretted as uniform weighting.

# New implementations

Overloading is optional. A fallback returns `nothing`.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.weights)"; overloaded=true))

"""
weights(::Any, data) = nothing

"""
    LearnAPI.features(algorithm, data)

Return, for each form of `data` supported in a call of the form `[`fit(algorithm,
data)`](@ref), the "features" part of `data` (as opposed to the target
variable, for example). 

The returned object `X` may always be passed to `predict` or `transform`, where
implemented, as in the following sample workflow:

```julia
model = fit(algorithm, data)
X = features(data)
ŷ = predict(algorithm, kind_of_proxy, X) # eg, `kind_of_proxy = LiteralTarget()`
```

The return value has the same number of observations as `data` does. For supervised models
(i.e., where `:(LearnAPI.target) in LearnAPI.functions(algorithm)`) `ŷ` above is generally
inteneded to be an approximate proxy for `LearnAPI.target(algorithm, data)`, the training
target.


# New implementations

The only contract `features` must satisfy is the one about passability of the output to
`predict` or `transform`, for each supported input `data`. The following fallbacks
typically make overloading `LearnAPI.features` unnecessary:

```julia
LearnAPI.features(algorithm, data) = data
LearnAPI.features(algorithm, data::Tuple) = first(data)
```

Overloading may be necessary if [`obs(algorithm, data)`](@ref) is overloaded to return
some algorithm-specific representation of training `data`. For density estimators, whose
`fit` typically consumes *only* a target variable, you should overload this method to
return `nothing`.

"""
features(algorithm, data) = data
features(algorithm, data::Tuple) = first(data)
