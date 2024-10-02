# [`obs` and Data Interfaces](@id data_interface)

The `obs` method takes data intended as input to `fit`, `predict` or `transform`, and
transforms it to an algorithm-specific form guaranteed to implement a form of observation
access designated by the algorithm. The transformed data can then be resampled and passed
on to the relevant method in place of the original input. Using `obs` may provide
performance advantages over naive workflows in some cases (e.g., cross-validation).

```julia
obs(algorithm, data) # can be passed to `fit` instead of `data`
obs(model, data)     # can be passed to `predict` or `transform` instead of `data`
```

## [Typical workflows](@id obs_workflows)

LearnAPI.jl makes no universal assumptions about the form of `data` in a call
like `fit(algorithm, data)`. However, if we define

```julia
observations = obs(algorithm, data)
```

then, assuming the typical case that `LearnAPI.data_interface(algorithm) ==
LearnAPI.RandomAccess()`, `observations` implements the
[MLUtils.jl](https://juliaml.github.io/MLUtils.jl/dev/) `getobs`/`numobs` interface, for
grabbing and counting observations. Moreover, we can pass `observations` to `fit` in place
of the original data, or first resample it using `MLUtils.getobs`:

```julia
# equivalent to `model = fit(algorithm, data)`
model = fit(algorithm, observations)

# with resampling:
resampled_observations = MLUtils.getobs(observations, 1:10)
model = fit(algorithm, resampled_observations)
```

In some implementations, the alternative pattern above can be used to avoid repeating
unnecessary internal data preprocessing, or inefficient resampling.  For example, here's
how a user might call `obs` and `MLUtils.getobs` to perform efficient cross-validation:

```julia
using LearnAPI
import MLUtils

algorithm = <some supervised learner>

data = <some data that `fit` can consume, with 30 observations>
X = LearnAPI.features(algorithm, data)
y = LearnAPI.target(algorithm, data)

train_test_folds = map([1:10, 11:20, 21:30]) do test
    (setdiff(1:30, test), test)
end

fitobs = obs(algorithm, data)
never_trained = true

scores = map(train_test_folds) do (train, test)

    # train using model-specific representation of data:
    fitobs_subset = MLUtils.getobs(fitobs, train)
    model = fit(algorithm, fitobs_subset)

    # predict on the fold complement:
    if never_trained
        global predictobs = obs(model, X)
        global never_trained = false
    end
    predictobs_subset = MLUtils.getobs(predictobs, test)
    ŷ = predict(model, Point(), predictobs_subset)

    return <score comparing ŷ with y[test]>

end
```

## Implementation guide

| method                                  | compulsory? | fallback       |
|:----------------------------------------|:-----------:|:--------------:|
| [`obs(algorithm_or_model, data)`](@ref) | depends     | returns `data` |
|                                         |             |                |

A sample implementation is given in [Providing an advanced data interface](@ref). 


## Reference

```@docs
obs
```

### [Data interfaces](@id data_interfaces)

New implementations must overload [`LearnAPI.data_interface(algorithm)`](@ref) if the
output of [`obs`](@ref) does not implement [`LearnAPI.RandomAccess`](@ref). (Arrays, most
tables, and all tuples thereof, implement `RandomAccess`.)

- [`LearnAPI.RandomAccess`](@ref) (default)
- [`LearnAPI.FiniteIterable`](@ref)
- [`LearnAPI.Iterable`](@ref)


```@docs
LearnAPI.RandomAccess
LearnAPI.FiniteIterable
LearnAPI.Iterable
```

