"""
    obs(learner, data)
    obs(model, data)

Return learner-specific representation of `data`, suitable for passing to `fit`
(first signature) or to `predict` and `transform` (second signature), in place of
`data`. Here `model` is the return value of `fit(learner, ...)` for some LearnAPI.jl
learner, `learner`.

The returned object is guaranteed to implement observation access as indicated by
[`LearnAPI.data_interface(learner)`](@ref), typically
[`LearnAPI.RandomAccess()`](@ref).

Calling `fit`/`predict`/`transform` on the returned objects may have performance
advantages over calling directly on `data` in some contexts. And resampling the returned
object using `MLUtils.getobs` may be cheaper than directly resampling the components of
`data`.

# Example

Usual workflow, using data-specific resampling methods:

```julia
data = (X, y) # a DataFrame and a vector
data_train = (Tables.select(X, 1:100), y[1:100])
model = fit(learner, data_train)
ŷ = predict(model, Point(), X[101:150])
```

Alternative, data agnostic, workflow using `obs` and the MLUtils.jl method `getobs`
(assumes `LearnAPI.data_interface(learner) == RandomAccess()`):

```julia
import MLUtils

fit_observations = obs(learner, data)
model = fit(learner, MLUtils.getobs(fit_observations, 1:100))

predict_observations = obs(model, X)
ẑ = predict(model, Point(), MLUtils.getobs(predict_observations, 101:150))
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
`predict(model, ..., observations)` and `transform(model, observations)` are supported
alternatives, whenever `observations = obs(model, data)`.

The fallback for `obs` is `obs(model_or_learner, data) = data`, and the fallback for
`LearnAPI.data_interface(learner)` is `LearnAPI.RandomAccess()`. For details refer to
the [`LearnAPI.data_interface`](@ref) document string.

In particular, if the `data` to be consumed by `fit`, `predict` or `transform` consists
only of suitable tables and arrays, then `obs` and `LearnAPI.data_interface` do not need
to be overloaded. However, the user will get no performance benefits by using `obs` in
that case.

When overloading `obs(learner, data)` to output new model-specific representations of
data, it may be necessary to also overload [`LearnAPI.features`](@ref),
[`LearnAPI.target`](@ref) (supervised learners), and/or [`LearnAPI.weights`](@ref) (if
weights are supported), for extracting relevant parts of the representation.

## Sample implementation

Refer to the "Anatomy of an Implementation" section of the LearnAPI.jl
[manual](https://juliaai.github.io/LearnAPI.jl/dev/).


"""
obs(learner_or_model, data) = data
