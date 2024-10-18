# [Accessor Functions](@id accessor_functions)

The sole argument of an accessor function is the output, `model`, of
[`fit`](@ref). Learners are free to implement any number of these, or none of them. Only
`LearnAPI.strip` has a fallback, namely the identity.

- [`LearnAPI.learner(model)`](@ref)
- [`LearnAPI.extras(model)`](@ref)
- [`LearnAPI.strip(model)`](@ref)
- [`LearnAPI.coefficients(model)`](@ref)
- [`LearnAPI.intercept(model)`](@ref)
- [`LearnAPI.tree(model)`](@ref)
- [`LearnAPI.trees(model)`](@ref)
- [`LearnAPI.feature_importances(model)`](@ref)
- [`LearnAPI.training_labels(model)`](@ref)
- [`LearnAPI.training_losses(model)`](@ref)
- [`LearnAPI.training_predictions(model)`](@ref)
- [`LearnAPI.training_scores(model)`](@ref)
- [`LearnAPI.components(model)`](@ref)

Learner-specific accessor functions may also be implemented. The names of all accessor
functions are included in the list returned by [`LearnAPI.functions(learner)`](@ref).

## Implementation guide

All new implementations must implement [`LearnAPI.learner`](@ref). While, all others are
optional, any implemented accessor functions must be added to the list returned by
[`LearnAPI.functions`](@ref).


## Reference

```@docs
LearnAPI.learner
LearnAPI.extras
LearnAPI.strip
LearnAPI.coefficients
LearnAPI.intercept
LearnAPI.tree
LearnAPI.trees
LearnAPI.feature_importances
LearnAPI.training_losses
LearnAPI.training_predictions
LearnAPI.training_scores
LearnAPI.training_labels
LearnAPI.components
```


