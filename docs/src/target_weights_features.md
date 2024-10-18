# [`target`, `weights`, and `features`](@id input)

Methods for extracting parts of training data:

```julia
LearnAPI.target(learner, data) -> <target variable>
LearnAPI.weights(learner, data) -> <per-observation weights>
LearnAPI.features(learner, data) -> <training "features", suitable input for `predict` or `transform`>
```

Here `data` is something supported in a call of the form `fit(learner, data)`. 

# Typical workflow

Not typically appearing in a general user's workflow but useful in meta-alagorithms, such
as cross-validation (see the example in [`obs` and Data Interfaces](@ref data_interface)).

Supposing `learner` is a supervised classifier predicting a one-dimensional vector
target:

```julia
model = fit(learner, data)
X = LearnAPI.features(learner, data)
y = LearnAPI.target(learner, data)
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
