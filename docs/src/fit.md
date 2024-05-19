# [`fit`](@ref fit)

```julia
fit(algorithm, data; verbosity=1) -> model
fit(model, data; verbosity=1) -> updated_model
```

When `fit` expects an tuple form of argument, `data = (X1, ..., Xn)`, then the signature
`fit(algorithm, X1, ..., Xn)` is also provided.

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

| method                    | fallback | compulsory? |
|:--------------------------|:---------|-------------|
| [`fit`](@ref)`(alg, ...)` | none     | yes         |



## Reference

```@docs
LearnAPI.fit
```
