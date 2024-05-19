"""
    obs(algorithm, data)
    obs(model, data)

Return an algorithm-specific representation of `data`, suitable for passing to `fit`
(first signature) or to `predict` and `transform` (second signature), in place of
`data`. Here `model` is the return value of `fit(algorithm, ...)` for some LearnAPI.jl
algorithm, `algorithm`.

The returned object is guaranteed to implement observation access as indicated
by [`LearnAPI.data_interface(algorithm)`](@ref) (typically the
[MLUtils.jl](https://juliaml.github.io/MLUtils.jl/dev/) `getobs`/`numobs` interface).

Calling `fit`/`predict`/`transform` on the returned objects may have performance
advantages over calling directly on `data` in some contexts. And resampling the returned
object using `MLUtils.getobs` may be cheaper than directly resampling the components of
`data`.

# Example

Usual workflow, using data-specific resampling methods:

```julia
X = <some `DataFrame`>
y = <some `Vector`>

Xtrain = Tables.select(X, 1:100)
ytrain = y[1:100]
model = fit(algorithm, (Xtrain, ytrain))
ŷ = predict(model, LiteralTarget(), y[101:150])
```

Alternative workflow using `obs` and the MLUtils.jl API:

```julia
import MLUtils

fit_obsevations = obs(algorithm, (X, y))
model = fit(algorithm, MLUtils.getobs(fit_observations, 1:100))

predict_observations = obs(model, X)
ẑ = predict(model, LiteralTarget(), MLUtils.getobs(predict_observations, 101:150))
@assert ẑ == ŷ
```

See also [`LearnAPI.data_interface`](@ref).


# Extended help

# New implementations

Implementation is typically optional.

For each supported form of `data` in `fit(algorithm, data)`, `predict(model, data)`, and
`transform(model, data)`, it must be true that `model = fit(algorithm, observations)` is
supported, whenever `observations = obs(algorithm, data)`, and that `predict(model,
observations)` and `transform(model, observations)` are supported, whenever `observations
= obs(model, data)`.

The fallback for `obs` is `obs(model_or_algorithm, data) = data`, and the fallback for
`LearnAPI.data_interface(algorithm)` indicates MLUtils.jl as the adopted interface. For
details refer to the [`LearnAPI.data_interface`](@ref) document string.

In particular, if the `data` to be consumed by `fit`, `predict` or `transform` consists
only of suitable tables and arrays, then `obs` and `LearnAPI.data_interface` do not need
to be overloaded. However, the user will get no performance benefits by using `obs` in
that case.

## Sample implementation

Refer to the "Anatomy of an Implemetation" section of the LearnAPI
[manual](https://juliaai.github.io/LearnAPI.jl/dev/).


"""
obs(algorithm_or_model, data) = data
