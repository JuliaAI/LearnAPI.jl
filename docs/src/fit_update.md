# [`fit`, `update`, `update_observations`, and `update_features`](@id fit_docs)

### Training

```julia
fit(learner, data; verbosity=LearnAPI.default_verbosity()) -> model
fit(learner; verbosity=LearnAPI.default_verbosity()) -> static_model 
```

A "static" algorithm is one that does not generalize to new observations (e.g., some
clustering algorithms); there is no training data and the algorithm is executed by
`predict` or `transform` which receive the data. See example below.


### Updating

```
update(model, data; verbosity=..., param1=new_value1, param2=new_value2, ...) -> updated_model
update_observations(model, new_data; verbosity=..., param1=new_value1, ...) -> updated_model
update_features(model, new_data; verbosity=..., param1=new_value1, ...) -> updated_model
```

## Typical workflows

### Supervised models

Supposing `Learner` is some supervised classifier type, with an iteration parameter `n`:

```julia
learner = Learner(n=100)
model = fit(learner, (X, y))

# Predict probability distributions:
ŷ = predict(model, Distribution(), Xnew) 

# Inspect some byproducts of training:
LearnAPI.feature_importances(model)

# Add 50 iterations and predict again:
model = update(model; n=150)
predict(model, Distribution(), X)
```

See also [Classification](@ref) and [Regression](@ref).

### Transformers

A dimension-reducing transformer, `learner`,  might be used in this way:

```julia
model = fit(learner, X)
transform(model, X) # or `transform(model, Xnew)`
```

or, if implemented, using a single call:

```julia
transform(learner, X) # `fit` implied
```

### [Static algorithms (no "learning")](@id static_algorithms)

Suppose `learner` is some clustering algorithm that cannot be generalized to new data
(e.g. DBSCAN):

```julia
model = fit(learner) # no training data
labels = predict(model, X) # may mutate `model`

# Or, in one line:
labels = predict(learner, X)

# But two-line version exposes byproducts of the clustering algorithm (e.g., outliers):
LearnAPI.extras(model)
```

See also [Static Algorithms](@ref)

### [Density estimation](@id density_estimation)

In density estimation, `fit` consumes no features, only a target variable; `predict`,
which consumes no data, returns the learned density:

```julia
model = fit(learner, y) # no features
predict(model)  # shortcut for  `predict(model, SingleDistribution())`, or similar
```

A one-liner will typically be implemented as well:

```julia
predict(learner, y)
```

See also [Density Estimation](@ref).


## Implementation guide

### Training

Exactly one of the following must be implemented:

| method                                                                 | fallback |
|:-----------------------------------------------------------------------|:---------|
| [`fit`](@ref)`(learner, data; verbosity=LearnAPI.default_verbosity())` | none     |
| [`fit`](@ref)`(learner; verbosity=LearnAPI.default_verbosity())`       | none     |

### Updating

| method                                                                               | fallback | compulsory? |
|:-------------------------------------------------------------------------------------|:---------|-------------|
| [`update`](@ref)`(model, data; verbosity=..., hyperparameter_updates...)`              | none     | no          |
| [`update_observations`](@ref)`(model, new_data; verbosity=..., hyperparameter_updates...)` | none     | no          |
| [`update_features`](@ref)`(model, new_data; verbosity=..., hyperparameter_updates...)`     | none     | no          |

There are some contracts governing the behaviour of the update methods, as they relate to
a previous `fit` call. Consult the document strings for details.

## Reference

```@docs
fit
update
update_observations
update_features
LearnAPI.default_verbosity
```
