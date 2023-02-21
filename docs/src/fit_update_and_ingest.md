# Fit, update! and ingest!

> **Summary.** Algorithms that learn, i.e., generalize to new data, must overload `fit`; the
> fallback performs no operation and returns all `nothing`. Implement `update!` if certain
> hyperparameter changes do not necessitate retraining from scratch (e.g., increasing an
> iteration parameter). Implement `ingest!` to implement incremental learning. All
> training methods implemented must be named in the return value of the
> `functions` trait.

| method                     | fallback                                           | compulsory? | requires          |
|:---------------------------|:---------------------------------------------------|-------------|-------------------|
| [`LearnAPI.fit`](@ref)     | does nothing, returns `(nothing, nothing, nothing)`| no          |                   |
| [`LearnAPI.update!`](@ref) | calls `fit`                                        | no          | [`LearnAPI.fit`](@ref) |
| [`LearnAPI.ingest!`](@ref) | none                                               | no          | [`LearnAPI.fit`](@ref) |

All three methods above return a triple `(fitted_params, state, report)` whose components
are explained under [`LearnAPI.fit`](@ref) below.  Items that might be returned in
`report` include: feature rankings/importances, SVM support vectors, clustering centers,
methods for visualizing training outcomes, methods for saving learned parameters in a
custom format, degrees of freedom, deviances. Precisely what `report` includes might be
controlled by hyperparameters (algorithm properties) especially if there is a performance
cost to it's inclusion.

Implement `fit` unless all [operations](@ref operations), such as `predict` and
`transform`, ignore their `fitted_params` argument (which will be `nothing`). This is the
case for many algorithms that have hyperparameters, but do not generalize to new data, such
as a basic DBSCAN clustering algorithm.

The `update!` method is intended for all subsequent calls to train an algorithm *using the same
observations*, but with possibly altered hyperparameters (`algorithm` argument). A fallback
implementation simply calls `fit`. The main use cases for implementing `update` are: 

- warm-restarting iterative algorithms

- "smart" training of composite algorithms, such as linear pipelines; here "smart" means that
  hyperparameter changes only trigger the retraining of downstream components.

The `ingest!` method supports incremental learning (same hyperparameters, but new training
observations). Like `update!`, it depends on the output a preceding `fit` or `ingest!`
call.

```@docs
LearnAPI.fit
LearnAPI.update!
LearnAPI.ingest!
```
