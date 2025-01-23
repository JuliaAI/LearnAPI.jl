# [`target`, `weights`, and `features`](@id input)

Methods for extracting parts of training observations. Here "observations" means the
output of [`obs(learner, data)`](@ref); if `obs` is not overloaded for `learner`, then
"observations" is any `data` supported in calls of the form [`fit(learner, data)`](@ref)

```julia
LearnAPI.target(learner, observations) -> <target variable>
LearnAPI.weights(learner, observations) -> <per-observation weights>
LearnAPI.features(learner, observations) -> <training "features", suitable input for `predict` or `transform`>
```

Here `data` is something supported in a call of the form `fit(learner, data)`. 

# Typical workflow

Not typically appearing in a general user's workflow but useful in meta-alagorithms, such
as cross-validation (see the example in [`obs` and Data Interfaces](@ref data_interface)).

Supposing `learner` is a supervised classifier predicting a one-dimensional vector
target:

```julia
observations = obs(learner, data)
model = fit(learner, observations)
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
