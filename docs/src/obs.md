# [`obs` and Data Interfaces](@id data_interface)

The `obs` method takes data intended as input to `fit`, `predict` or `transform`, and
transforms it to a learner-specific form guaranteed to implement a form of observation
access designated by the learner.  The transformed data can then passed on to the relevant
method in place of the original input (after first resampling it, if the learner supports
this). Using `obs` may provide performance advantages over naive workflows in some cases
(e.g., cross-validation).

```julia
obs(learner, data) # can be passed to `fit` instead of `data`
obs(model, data)   # can be passed to `predict` or `transform` instead of `data`
```

- [Data interfaces](@ref data_interfaces)


## [Typical workflows](@id obs_workflows)

LearnAPI.jl makes no universal assumptions about the form of `data` in a call
like `fit(learner, data)`. However, if we define

```julia
observations = obs(learner, data)
```

then, assuming the typical case that `LearnAPI.data_interface(learner) ==
LearnAPI.RandomAccess()`, `observations` implements the
[MLCore.jl](https://juliaml.github.io/MLCore.jl/dev/) `getobs`/`numobs` interface, for
grabbing and counting observations. Moreover, we can pass `observations` to `fit` in place
of the original data, or first resample it using `MLCore.getobs`:

```julia
# equivalent to `model = fit(learner, data)`
model = fit(learner, observations)

# with resampling:
resampled_observations = MLCore.getobs(observations, 1:10)
model = fit(learner, resampled_observations)
```

In some implementations, the alternative pattern above can be used to avoid repeating
unnecessary internal data preprocessing, or inefficient resampling.  For example, here's
how a user might call `obs` and `MLCore.getobs` to perform efficient cross-validation:

```julia
using LearnAPI
import MLCore

learner = <some supervised learner>

data = <some data that `fit` can consume, with 30 observations>

train_test_folds = map([1:10, 11:20, 21:30]) do test
    (setdiff(1:30, test), test)
end

fitobs = obs(learner, data)
never_trained = true

scores = map(train_test_folds) do (train, test)

    # train using model-specific representation of data:
    fitobs_subset = MLCore.getobs(fitobs, train)
    model = fit(learner, fitobs_subset)

    # predict on the fold complement:
    if never_trained
        X = LearnAPI.features(learner, data)
        global predictobs = obs(model, X)
        global never_trained = false
    end
    predictobs_subset = MLCore.getobs(predictobs, test)
    ŷ = predict(model, Point(), predictobs_subset)

    y = LearnAPI.target(learner, data)
    return <score comparing ŷ with y[test]>

end
```

## Implementation guide

| method                         | comment                             | compulsory?   | fallback       |
|:-------------------------------|:------------------------------------|:-------------:|:---------------|
| [`obs(learner, data)`](@ref) | here `data` is `fit`-consumable     | not typically | returns `data` |
| [`obs(model, data)`](@ref)     | here `data` is `predict`-consumable | not typically | returns `data` |


A sample implementation is given in [Providing a separate data front end](@ref). 


## Reference

```@docs
obs
```

### [Available data interfaces](@id data_interfaces)


```@docs
LearnAPI.DataInterface
LearnAPI.RandomAccess
LearnAPI.FiniteIterable
LearnAPI.Iterable
```

