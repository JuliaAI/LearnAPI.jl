# [Accessor Functions](@id accessor_functions)

The sole argument of an accessor function is the output, `model`, of [`fit`](@ref) or
[`obsfit`](@ref).

- [`LearnAPI.algorithm(model)`](@ref)
- [`LearnAPI.extras(model)`](@ref)
- [`LearnAPI.coefficients(model)`](@ref)
- [`LearnAPI.intercept(model)`](@ref)
- [`LearnAPI.tree(model)`](@ref)
- [`LearnAPI.trees(model)`](@ref)
- [`LearnAPI.feature_importances(model)`](@ref)
- [`LearnAPI.training_labels(model)`](@ref)
- [`LearnAPI.training_losses(model)`](@ref)
- [`LearnAPI.training_scores(model)`](@ref)
- [`LearnAPI.components(model)`](@ref)

## Implementation guide

All new implementations must implement [`LearnAPI.algorithm`](@ref). All others are
optional.

## Reference

```@docs
LearnAPI.algorithm
LearnAPI.extras
LearnAPI.coefficients
LearnAPI.intercept
LearnAPI.tree
LearnAPI.trees
LearnAPI.feature_importances
LearnAPI.training_losses
LearnAPI.training_scores
LearnAPI.training_labels
LearnAPI.components
```


