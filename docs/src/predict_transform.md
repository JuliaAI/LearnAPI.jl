# [`predict`, `transform` and `inverse_transform`](@id operations)

```julia
predict(model, kind_of_proxy, data)
transform(model, data)
inverse_transform(model, data)
```

Versions without the `data` argument may apply, for example in [Density
estimation](@ref).

## [Typical worklows](@id predict_workflow)

Train some supervised `learner`:

```julia
model = fit(learner, (X, y))
```

Predict probability distributions:

```julia
ŷ = predict(model, Distribution(), Xnew)
```

Generate point predictions:

```julia
ŷ = predict(model, Point(), Xnew)
```

Train a dimension-reducing `learner`:

```julia
model = fit(learner, X)
Xnew_reduced = transform(model, Xnew)
```

Apply an approximate right inverse:

```julia
inverse_transform(model, Xnew_reduced)
```

Fit and transform in one line:

```julia
transform(learner, data) # `fit` implied
```

### An advanced workflow

```julia
fitobs = obs(learner, (X, y)) # learner-specific repr. of data
model = fit(learner, MLUtils.getobs(fitobs, 1:100))
predictobs = obs(model, MLUtils.getobs(X, 101:150))
ŷ = predict(model, Point(), predictobs)
```


## [Implementation guide](@id predict_guide)

| method                      | compulsory? | fallback |
|:----------------------------|:-----------:|:--------:|
| [`predict`](@ref)           | no          | none     |
| [`transform`](@ref)         | no          | none     |
| [`inverse_transform`](@ref) | no          | none     |

### Predict or transform?

If the learner has a notion of [target variable](@ref proxy), then use 
[`predict`](@ref) to output each supported [kind of target proxy](@ref
proxy_types) (`Point()`, `Distribution()`, etc).

For output not associated with a target variable, implement [`transform`](@ref)
instead, which does not dispatch on [`LearnAPI.KindOfProxy`](@ref), but can be optionally
paired with an implementation of [`inverse_transform`](@ref), for returning (approximate)
right or left inverses to `transform`.

Of course, the one learner can implement both a `predict` and `transform` method. For
example a K-means clustering algorithm can `predict` labels and `transform` to reduce
dimension using distances from the cluster centres.


### [One-liners combining fit and transform/predict](@id one_liners)

Learners may optionally overload `transform` to apply `fit` first, using the supplied
data if required, and then immediately `transform` the same data. The same applies to
`predict`. In that case the first argument of `transform`/`predict` is an *learner*
instead of the output of `fit`:

```julia
predict(learner, kind_of_proxy, data) # `fit` implied
transform(learner, data) # `fit` implied
```

For example, if `fit(learner, X)` is defined, then `predict(learner, X)` will be
shorthand for

```julia
model = fit(learner, X)
predict(model, X)
```

## [Reference](@id predict_ref)

```@docs
predict
transform
inverse_transform
```
