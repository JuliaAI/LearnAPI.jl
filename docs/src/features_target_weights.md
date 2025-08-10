# [`features`, `target`, and `weights`](@id input)

Methods for extracting certain parts of `data` for all supported calls of the form
[`fit(learner, data)`](@ref).

```julia
LearnAPI.features(learner, data) -> <training "features"; suitable input for `predict` or `transform`>
LearnAPI.target(learner, data) -> <target variable>
LearnAPI.weights(learner, data) -> <per-observation weights>
```

Here `data` is something supported in a call of the form `fit(learner, data)`. 

# Typical workflow

Not typically appearing in a general user's workflow but useful in meta-alagorithms, such
as cross-validation (see the example in [`obs` and Data Interfaces](@ref data_interface)).

Supposing `learner` is a supervised classifier predicting a vector
target:

```julia
model = fit(learner, data)
X = LearnAPI.features(learner, data)
y = LearnAPI.target(learner, data)
ŷ = predict(model, Point(), X)
training_loss = sum(ŷ .!= y)
```

# Implementation guide

| method                                     | fallback return value | compulsory? |
|:-------------------------------------------|:---------------------:|-------------|
| [`LearnAPI.features(learner, data)`](@ref) | no fallback           | no          |
| [`LearnAPI.target(learner, data)`](@ref)   | no fallback           | no          |
| [`LearnAPI.weights(learner, data)`](@ref)  | `nothing`             | no          |
 

# Reference

```@docs
LearnAPI.features
LearnAPI.target
LearnAPI.weights
```
