# [`obs`](@id data_interface)

The `obs` method takes data intended as input to `fit`, `predict` or `transform`, and
transforms it to an algorithm-specific form guaranteed to implement a form of observation
access designated by the algorithm. The transformed data can then be resampled and passed
on to the relevant method in place of the original input. Using `obs` may provide
performance advantages over naive workflows in some cases (e.g., cross-validation).

```julia
obs(algorithm, data) # can be passed to `fit` instead of `data`
obs(model, data)     # can be passed to `predict` or `transform` instead of `data`
```

## Typical workflows

LearnAPI.jl makes no explicit assumptions about the form of data `X` and `y` in a call
like `fit(algorithm, (X, y))`. However, if we define

```julia
observations = obs(algorithm, (X, y))
```

then, assuming the typical case that `LearnAPI.data_interface(algorithm) == Base.HasLength()`, `observations` implements the [MLUtils.jl](https://juliaml.github.io/MLUtils.jl/dev/) `getobs`/`numobs` interface. Moreover, we can pass `observations` to `fit` in place of
the original data, or first resample it using `MLUtils.getobs`:

```julia
# equivalent to `model = fit(algorithm, (X, y))` (or `fit(algorithm, X, y))`:
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

X = <some data frame with 30 rows>
y = <some categorical vector with 30 rows>
algorithm = <some LearnAPI-compliant algorithm>

train_test_folds = map([1:10, 11:20, 21:30]) do test
    (setdiff(1:30, test), test)
end

fitobs = obs(algorithm, (X, y))
never_trained = true

scores = map(train_test_folds) do (train, test)

    # train using model-specific representation of data:
    trainobs = MLUtils.getobs(fitobs, train)
    model = fit(algorithm, trainobs)

    # predict on the fold complement:
    if never_trained
        global predictobs = obs(model, X)
        global never_trained = false
    end
    testobs = MLUtils.getobs(predictobs, test)
    ŷ = predict(model, LiteralTarget(), testobs)

    return <score comparing ŷ with y[test]>

end
```

Note here that the output of `predict` will match the representation of `y` , i.e.,
there is no concept of an algorithm-specific representation of *outputs*, only inputs.


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