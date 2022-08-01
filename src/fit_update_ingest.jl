const DOC_OPERATIONS =
    "An *operation* is a method like [`MLInterface.predict`](@ref) or "*
    "[`MLInterface.transform`](@ref); do `MLInterface.OPERATIONS` to list."

DOC_IMPLEMENTED_METHODS(name) =
    "If implemented, include `:$name` in the vector returned by the "*
    "[`MLInterface.implemented_methods`](@ref) trait. "


# # FIT

"""
    MLInterface.fit(model, verbosity, data...; metadata...)

Fit `model` to the provided `data` and `metadata`. With the exception of warnings, training
should be silent if `verbosity == 0`. Lower values should suppress warnings. Here:

- `model` is a property-accessible object whose properties are the hyper-parameters of some
   machine learning algorithm; see also [`MLInterface.ismodel`](@ref).

- `data` is a tuple of data objects with a common number of observations, for example,
  `data = (X, y, w)` where `X` is a table of features, `y` a target variable, and `w`
  per-observation weights. The ML Model Interface does not specify how observations are
  structured or accessed.

- `metadata` is for extra information pertaining to the data that is not structured as a
  number of observations, for example, weights for target classes. Another example
  would be feature groupings in the group lasso algorithm.


# Return value

Returns a tuple (`fitresult`, `state`, `report`) where:

- The `fitresult` is the model's learned parameters (eg, the coefficients in a linear model)
  in a form understood by model operations. $DOC_OPERATIONS If some training outcome of
  user-interest is not needed for operations, it should be part of `report` instead (see
  below).

- The `state` is for passing to [`MLInterface.update!`](@ref) or
  [`MLInterface.ingest!`](@ref). For models that implement neither, `state` should be
  `nothing`.

- The `report` records byproducts of training not in the `fitresult`.


# New model implementations

This method is an optional method  the ML Model Interface. A fallback performs no
computation, returning `(nothing, nothing, nothing)`.

$(DOC_IMPLEMENTED_METHODS(:fit))

See also [`MLInterface.update!`](@ref), [`MLInterface.ingest!`](@ref).

"""
fit(::Any, ::Any, ::Integer, data...; metadata...) = nothing, nothing, nothing


# # UPDATE

"""
    MLInterface.update!(model, verbosity, fitresult, state, data...; metadata...)d Based on the values of `state`, and `fitresult` returned by a preceding call to
[`MLInterface.fit`](ref), [`MLInterface.ingest!`](@ref), or [`MLInterface.update!`](@ref),
update a model's learned parameters, returning new (or mutated) `state` and `fitresult`.

Intended for retraining a model when the training data has not changed, but `model`
properties (hyperparameters) may have changed. Specifically, the assumption is that `data`
and `metadata` have the same values seen in the most recent call to `fit/update!/ingest!`.

The most common use case is for continuing the training of an iterative model: `state` is
simply a copy of the model used in the last training call (`fit`, `update!' or `ingest!`) and
this will include the current number of iterations as a property. If `model` and `state`
differ only in the number of iterations (e.g., epochs in a neural networ), which has
increased, then the learned parameters (weights) are updated, rather computed ab
initio. Otherwise, `update!` simply calls `fit` to retrain from scratch.

**Important.** It is permitted to return mutated versions of `state` and `fitresult`, rather
  than new objects, but no other argument may be mutated.

For incremental training (same model, new data) see instead [`MLInterface.ingest!`](@ref).


# Return value

Same as [`MLInterface.fit`](@ref), namely a tuple (`fitresult`, `state`, `report`). See
[`MLInterface.fit`](@ref) for details.


# New model implementations

This method is an optional method in the ML Model Interface. A fallback calls
`MLInterfaceperforms.fit`:

```julia
MLInterface.update!(model, verbosity, fitresult, state, data...; metadata...) =
    fit(model, verbosity, data; metadata...)
```

$(DOC_IMPLEMENTED_METHODS(:fit))

See also [`MLInterface.fit`](@ref), [`MLInterface.ingest!`](@ref).

"""
update!(model, verbosity, fitresult, state, data...; metadata...) =
    fit(model, verbosity, data...; metadata...)


# # INGEST

"""
    MLInterface.ingest!(model, verbosity, fitresult, state, data...; metadata...)

For a model that supports incremental learning, update the learned parameters using `data`,
which has typically not been seen before.  The arguments `state` and `fitresult` are the
output of a preceding call to [`MLInterface.fit`](ref), [`MLInterface.ingest!`](@ref), or
[`MLInterface.update!`](@ref), of which mutated or new versions are returned.

For updating learned parameters using the *same* data but new hyperparameters, see instead
[`MLInterface.update!`](@ref).

**Important.** It is permitted to return mutated versions of `state` and `fitresult`, rather
than new objects, but no other argument may be mutated.

For incremental training, see instead [`MLInterface.ingest'](@ref).


# Return value

Same as [`MLInterface.fit`](@ref), namely a tuple (`fitresult`, `state`, `report`). See
[`MLInterface.fit`](@ref) for details.


# New model implementations

This method is an optional method in the ML Model Interface. It has no fallback.

$(DOC_IMPLEMENTED_METHODS(:fit))

See also [`MLInterface.fit`](@ref), [`MLInterface.update!`](@ref).

"""
function ingest!(model, verbosity, fitresult, state, data...; metadata...) end
