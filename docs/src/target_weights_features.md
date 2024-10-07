# [`target`, `weights`, and `features`](@id input)

Methods for extracting parts of training data:

```julia
LearnAPI.target(algorithm, data) -> <target variable>
LearnAPI.weights(algorithm, data) -> <per-observation weights>
LearnAPI.features(algorithm, data) -> <training "features", suitable input for `predict` or `transform`>
```

Here `data` is something supported in a call of the form `fit(algorithm, data)`. 

# Typical workflow

Not typically appearing in a general user's workflow but useful in meta-alagorithms, such
as cross-validation (see the example in [`obs` and Data Interfaces](@ref data_interface)).

Supposing `algorithm` is a supervised classifier predicting a one-dimensional vector
target:

```julia
model = fit(algorithm, data)
X = LearnAPI.features(algorithm, data)
y = LearnAPI.target(algorithm, data)
ŷ = predict(model, Point(), X)
training_loss = sum(ŷ .!= y)
```

# Implementation guide

| method                      | fallback          | compulsory?              |
|:----------------------------|:-----------------:|--------------------------|
| [`LearnAPI.target`](@ref)   | returns `nothing` | no                       |
| [`LearnAPI.weights`](@ref)  | returns `nothing` | no                       |
| [`LearnAPI.features`](@ref) | see docstring     | if fallback insufficient |


# Reference

```@docs
LearnAPI.target
LearnAPI.weights
LearnAPI.features
```
