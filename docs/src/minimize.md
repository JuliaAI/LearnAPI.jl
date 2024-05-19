# [`minimize`](@id algorithm_minimize)

```julia
minimize(model) -> <smaller version of model suitable for serialization>
```

# Typical workflow

```julia
model = fit(algorithm, X, y)
ŷ = predict(model, LiteralTarget(), Xnew)
LearnAPI.feature_importances(model)

small_model = minimize(model)
serialize("my_model.jls", small_model)

recovered_model = deserialize("my_random_forest.jls")
@assert predict(recovered_model, LiteralTarget(), Xnew) == ŷ

# throws MethodError:
LearnAPI.feature_importances(recovered_model)
```

# Implementation guide

| method                       | compulsory? | fallback |
|:-----------------------------|:-----------:|:--------:|
| [`minimize`](@ref)           | no          | identity |

# Reference

```@docs
minimize
```
