"""
    obs(learner, data)
    obs(model, data)

Return learner-specific representation of `data`, suitable for passing to `fit`, `update`,
 `update_observations`, or `update_features` (first signature) or to `predict` and
 `transform` (second signature), in place of `data`. Here `model` is the return value of
 `fit(learner, ...)` for some LearnAPI.jl learner, `learner`.

The returned object is guaranteed to implement observation access as indicated by
[`LearnAPI.data_interface(learner)`](@ref), typically
[`LearnAPI.RandomAccess()`](@ref).

Calling `fit`/`predict`/`transform` on the returned objects may have performance
advantages over calling directly on `data` in some contexts.

# Example

Usual workflow, using data-specific resampling methods:

```julia
data = (X, y) # a DataFrame and a vector
data_train = (Tables.select(X, 1:100), y[1:100])
model = fit(learner, data_train)
ŷ = predict(model, Point(), X[101:150])
```

Alternative workflow using `obs` and the MLCore.jl method `getobs` to carry out
subsampling (assumes `LearnAPI.data_interface(learner) == RandomAccess()`):

```julia
import MLCore
fit_observations = obs(learner, data)
model = fit(learner, MLCore.getobs(fit_observations, 1:100))
predict_observations = obs(model, X)
ẑ = predict(model, Point(), MLCore.getobs(predict_observations, 101:150))
@assert ẑ == ŷ
```

See also [`LearnAPI.data_interface`](@ref).

# Extended help

# New implementations

Implementation is typically optional.

For each supported form of `data` in `fit(learner, data)`, it must be true that `model =
fit(learner, observations)` is equivalent to `model = fit(learner, data)`, whenever
`observations = obs(learner, data)`. For each supported form of `data` in calls
`predict(model, ..., data)` and `transform(model, data)`, where implemented, the calls
`predict(model, ..., observations)` and `transform(model, observations)` must be supported
alternatives with the same output, whenever `observations = obs(model, data)`.

If `LearnAPI.data_interface(learner) == RandomAccess()` (the default), then `fit`,
`predict` and `transform` must additionally accept `obs` output that has been *subsampled*
using `MLCore.getobs`, with the obvious interpretation applying to the outcomes of such
calls (e.g., if *all* observations are subsampled, then outcomes should be the same as if
using the original data).

It is required that `obs(learner, _)` and `obs(model, _)` are involutive, meaning both the
following hold:

```julia
obs(learner, obs(learner, data)) == obs(learner, data)
obs(model, obs(model, data) == obs(model, obs(model, data)
```

If one overloads `obs`, one typically needs additionally overloadings to guarantee
involutivity.

The fallback for `obs` is `obs(model_or_learner, data) = data`, and the fallback for
`LearnAPI.data_interface(learner)` is `LearnAPI.RandomAccess()`. For details refer to
the [`LearnAPI.data_interface`](@ref) document string.

In particular, if the `data` to be consumed by `fit`, `predict` or `transform` consists
only of suitable tables and arrays, then `obs` and `LearnAPI.data_interface` do not need
to be overloaded. However, the user will get no performance benefits by using `obs` in
that case.

## Sample implementation

Refer to the ["Anatomy of an
Implementation"](https://juliaai.github.io/LearnAPI.jl/dev/anatomy_of_an_implementation/#Providing-an-advanced-data-interface)
section of the LearnAPI.jl manual.


"""
obs(learner_or_model, data) = data
