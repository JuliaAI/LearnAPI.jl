# [`fit`](@ref fit)

```julia
fit(algorithm, data...; verbosity=1) -> model
fit(model, data...; verbosity=1) -> updated_model
```

## Typical workflow

```julia
# Train some supervised `algorithm`:
model = fit(algorithm, X, y)

# Predict probability distributions:
ŷ = predict(model, Distribution(), Xnew)

# Inspect some byproducts of training:
LearnAPI.feature_importances(model)
```

## Implementation guide

The `fit` method is not implemented directly. Instead, implement [`obsfit`](@ref).

| method                       | fallback | compulsory? | requires                    |
|:-----------------------------|:---------|-------------|-----------------------------|
| [`obsfit`](@ref)`(alg, ...)` | none     | yes         | [`obs`](@ref) in some cases |
|                              |          |             |                             |


## Reference

```@docs
LearnAPI.fit
LearnAPI.obsfit
```
