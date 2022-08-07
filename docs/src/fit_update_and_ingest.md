# Fit, update! and ingest!

> **Summary.** All models that learn, i.e., generalize to new data, must implement `fit`;
> the fallback, useful for so-called **static** models, performs no operation and returns
> all `nothing`. Implement `update!` if certain hyper-parameter changes do not necessitate
> retraining from scratch (e.g., iterative models). Implement `ingest!` to implement
> incremental learning.

| method                     | fallback                                           | compulsory? | requires          |
|:---------------------------|:---------------------------------------------------|-------------|-------------------|
[`MLInterface.fit`](@ref)    | does nothing, returns `(nothing, nothing, nothing)`| no          |                   |
[`MLInterface.update!`](@ref) | calls `fit`                                       | no          | `MLInterface.fit` |
[`MLJInterface.ingest!`](@ref)| none                                              | no          | `MLInterfac.fit`  |

Implement `fit` unless your model is **static**, meaning its [operations](@ref operations)
such as `predict` and `transform`, ignore their `fitted_params` argument (which will be
`nothing`). This is the case for models that have hyper-parameters, but do not generalize to
new data, such as a basic DBSCAN clustering algorithm. Related:
[`MLInterface.reporting_operations`](@ref), [Static Models](@ref).

The `update!` method is intended for all subsequent calls to train a model *using the same
data*, but with possibly altered hyperparameters (`model` argument). As described below, a
fallback implementation simply calls `fit`. The main use cases are for warm-restarting
iterative model training, and for "smart" training of composite models, such as linear
pipelines. Here "smart" means that hyperparameter changes only trigger the retraining of
downstream components.

The `ingest!` method supports incremental learning (same hyperparameters, but new
data). Like `update!`, it depends on the output a preceding `fit` or `ingest!` call.


```@docs
MLInterface.fit
MLInterface.update!
MLInterface.ingest!
```

## Further guidance on what goes where

Recall that the `fitted_params` returned as part of `fit` represents everything needed by an
[operation](@ref operations), such as [`MLInterface.predict`](@ref). 

The properties of your model (typically struct fields) are *hyperparameters*, i.e., those
parameters declared by the user ahead of time that generally affect the outcome of training
and are not learned. It is okay to add "control" parameters (such a specifying whether or
not to use a GPU). Use `report` to return *everything else*.  This includes: feature
rankings/importances, SVM support vectors, clustering centres, methods for visualizing
training outcomes, methods for saving learned parameters in a custom format, degrees of
freedom, deviances. If there is a performance cost to extra functionality you want to
expose, the functionality can be toggled on/off through a hyperparameter, but this should
otherwise be avoided.
