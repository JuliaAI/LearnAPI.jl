# [`fit`, `update`, `update_observations`, and `update_features`](@id fit)

### Training

```julia
fit(algorithm, data; verbosity=1) -> model
fit(algorithm; verbosity=1) -> static_model 
```

A "static" algorithm is one that does not generalize to new observations (e.g., some
clustering algorithms); there is no trainiing data and the algorithm is executed by
`predict` or `transform` which receive the data. See example below.

When `fit` expects a tuple form of argument, `data = (X1, ..., Xn)`, then the signature
`fit(algorithm, X1, ..., Xn)` is also provided.

### Updating

```
update(model, data; verbosity=1, param1=new_value1, param2=new_value2, ...) -> updated_model
update_observations(model, new_data; verbosity=1, param1=new_value1, ...) -> updated_model
update_features(model, new_data; verbosity=1, param1=new_value1, ...) -> updated_model
```

Data slurping forms are similarly provided for updating methods.

## Typical workflows

### Supervised models

Supposing `Algorithm` is some supervised classifier type, with an iteration parameter `n`:

```julia
algorithm = Algorithm(n=100)
model = fit(algorithm, (X, y)) # or `fit(algorithm, X, y)`

# Predict probability distributions:
ŷ = predict(model, Distribution(), Xnew) 

# Inspect some byproducts of training:
LearnAPI.feature_importances(model)

# Add 50 iterations and predict again:
model = update(model; n=150)
predict(model, Distribution(), X)
```

### Tranformers

A dimension-reducing transformer, `algorithm`  might be used in this way:

```julia
model = fit(algorithm, X)
transform(model, X) # or `transform(model, Xnew)`
```

or, if implemented, using a single call:

```julia
transform(algorithm, X) # `fit` implied
```

### Static algorithms (no "learning")

Suppose `algorithm` is some clustering algorithm that cannot be generalized to new data
(e.g. DBSCAN):

```julia
model = fit(algorithm) # no training data
labels = predict(model, X) # may mutate `model`

# Or, in one line:
labels = predict(algorithm, X)

# But two-line version exposes byproducts of the clustering algorithm (e.g., outliers):
LearnAPI.extras(model)
```

## Implementation guide

### Training

| method                                                                         | fallback                                                         | compulsory?        |
|:-------------------------------------------------------------------------------|:-----------------------------------------------------------------|--------------------|
| [`fit`](@ref)`(algorithm, data; verbosity=1)`                                  | ignores `data` and applies signature below                       | yes, unless static |
| [`fit`](@ref)`(algorithm; verbosity=1)`                                        | none                                                             | no, unless static  |

### Updating

| method                                                                               | fallback | compulsory? |
|:-------------------------------------------------------------------------------------|:---------|-------------|
| [`update`](@ref)`(model, data; verbosity=1, hyperparameter_updates...)`              | none     | no          |
| [`update_observations`](@ref)`(model, data; verbosity=1, hyperparameter_updates...)` | none     | no          |
| [`update_features`](@ref)`(model, data; verbosity=1, hyperparameter_updates...)`     | none     | no          |

There are some contracts regarding the behaviour of the update methods, as they relate to
a previous `fit` call. Consult the document strings for details.

## Reference

```@docs
fit
update
update_observations
update_features
```
