# [`predict`, `transform`, and relatives](@id operations)

Standard methods:

```julia
predict(model, kind_of_proxy, data...) -> prediction
transform(model, data...) -> transformed_data
inverse_transform(model, data...) -> inverted_data
```

Methods consuming output `obsdata` of data-preprocessor [`obs`](@ref):

```julia
obspredict(model, kind_of_proxy, obsdata) -> prediction
obstransform(model, obsdata) -> transformed_data
```

## Typical worklows

```julia
# Train some supervised `algorithm`:
model = fit(algorithm, X, y)

# Predict probability distributions:
ŷ = predict(model, Distribution(), Xnew)

# Generate point predictions:
ŷ = predict(model, LiteralTarget(), Xnew)
```

```julia
# Training a dimension-reducing `algorithm`:
model = fit(algorithm, X)
Xnew_reduced = transform(model, Xnew)

# Apply an approximate right inverse:
inverse_transform(model, Xnew_reduced)
```

### An advanced workflow

```julia
fitdata = obs(fit, algorithm, X, y)
predictdata = obs(predict, algorithm, Xnew)
model = obsfit(algorithm, obsdata)
ŷ = obspredict(model, LiteralTarget(), predictdata)
```


## Implementation guide

The methods `predict` and `transform` are not directly overloaded. Implement `obspredict`
and `obstransform` instead:

| method                      | compulsory? | fallback | requires                              |
|:----------------------------|:-----------:|:--------:|:-------------------------------------:|
| [`obspredict`](@ref)        | no          | none     | [`fit`](@ref)                         |
| [`obstransform`](@ref)      | no          | none     | [`fit`](@ref)                         |
| [`inverse_transform`](@ref) | no          | none     | [`fit`](@ref), [`obstransform`](@ref) |

### Predict or transform?

If the algorithm has a notion of [target variable](@ref proxy), then arrange for
[`obspredict`](@ref) to output each supported [kind of target proxy](@ref
proxy_types) (`LiteralTarget()`, `Distribution()`, etc).

For output not associated with a target variable, implement [`obstransform`](@ref)
instead, which does not dispatch on [`LearnAPI.KindOfProxy`](@ref), but can be optionally
paired with an implementation of [`inverse_transform`](@ref) for returning (approximate)
right inverses to `transform`.


## Reference

```@docs
predict
obspredict
transform
obstransform
inverse_transform
```
