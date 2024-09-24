# [`fit`](@ref fit)

Training for the first time:

```julia
fit(algorithm, data; verbosity=1) -> model
fit(algorithm; verbosity=1) -> static_model 
```

Updating:

```
fit(model, data; verbosity=1, param1=new_value1, param2=new_value2, ...) -> updated_model
fit(model, NewObservations(), new_data; verbosity=1, param1=new_value1, ...) -> updated_model
fit(model, NewFeatures(), new_data; verbosity=1, param1=new_value1, ...) -> updated_model
```

When `fit` expects a tuple form of argument, `data = (X1, ..., Xn)`, then the signature
`fit(algorithm, X1, ..., Xn)` is also provided. 

## Typical workflows

Supposing `Algorithm` is some supervised classifier type, with an iteration parameter `n`:

```julia
algorithm = Algorithm(n=100)
model = fit(algorithm, (X, y)) # or `fit(algorithm, X, y)`

# Predict probability distributions:
ŷ = predict(model, Distribution(), Xnew)

# Inspect some byproducts of training:
LearnAPI.feature_importances(model)

# Add 50 iterations and predict again:
model = fit(model; n=150)
predict(model, Distribution(), X)
```

### A static algorithm (no "learning")

```julia
# Apply some clustering algorithm which cannot be generalized to new data:
model = fit(algorithm)
labels = predict(model, LabelAmbiguous(), X) # mutates `model`

# inspect byproducts of the clustering algorithm (e.g., outliers):
LearnAPI.extras(model)
```

## Implementation guide

Initial training: 

| method                                                                         | fallback                                                         | compulsory?        |
|:-------------------------------------------------------------------------------|:-----------------------------------------------------------------|--------------------|
| [`fit`](@ref)`(algorithm, data; verbosity=1)`                                  | ignores `data` and applies signature below                       | yes, unless static |
| [`fit`](@ref)`(algorithm; verbosity=1)`                                        | none                                                             | no, unless static  |

Updating:

| method                                                                         | fallback                                                                   | compulsory? |
|:-------------------------------------------------------------------------------|:---------------------------------------------------------------------------|-------------|
| [`fit`](@ref)`(model, data; verbosity=1, param_updates...)`                    | retrains from scratch on `data` with specified hyperparameter replacements | no          |
| [`fit`](@ref)`(model, ::NewObservations, data; verbosity=1, param_updates...)` | none                                                                       | no          |
| [`fit`](@ref)`(model, ::NewFeatures, data; verbosity=1, param_updates...)`     | none                                                                       | no          |

## Reference

```@docs
LearnAPI.fit
```
