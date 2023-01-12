# # DOC STRING HELPERS

const TRAINING_FUNCTIONS = (:fit, :update!, :ingest!)

const DOC_OPERATIONS =
    "An *operation* is a method, like [`LearnAPI.predict`](@ref) or "*
    "[`LearnAPI.transform`](@ref), that has signature "*
    "`(model, fitted_params, data....)`; do `LearnAPI.OPERATIONS` to list."

function DOC_IMPLEMENTED_METHODS(name; overloaded=false)
    word = overloaded ? "overloaded" : "implemented"
    "If $word, you must include `:$name` in the tuple returned by the "*
    "[`LearnAPI.functions`](@ref) trait. "
end

const DOC_METADATA =
    "`metadata` is for extra information pertaining to the data that is never "*
    "iterated or subsampled, eg., weights for target classes. Another example "*
    "would be feature groupings "*
    "in the group lasso algorithm. "

const DOC_WHAT_IS_DATA =
    """
    Note that in LearnAPI.jl the word "data" is only defined informally, as an object
    generating "observations", which are not defined at all.
    """

const DOC_MUTATING_MODELS =
    """
    !!! note

        The method is not permitted to mutate `model`. In particular, if `model` has a
        random number generator as a hyperparameter (property) then it must be copied
        before use.  """

# # FIT

"""
    LearnAPI.fit(model, verbosity, data...; metadata...)

Fit `model` to the provided `data` and `metadata`. With the exception of warnings,
training will be silent if `verbosity == 0`. Lower values should suppress warnings; any
integer ought to be admissible. Here:

- `model` is a property-accessible object whose properties are the hyper-parameters of some
   machine learning algorithm.

- `data` is a tuple of data objects with a common number of observations, for example,
  `data = (X, y, w)` where `X` is a table of features, `y` is a target vector with the
  same number of rows, and `w` a vector of per-observation weights.

- $DOC_METADATA To see the keyword names for metadata supported by `model`, do
  `LearnAPI.fit_keywords(model)`. "


# Return value

Returns a tuple (`fitted_params`, `state`, `report`) where:

- The `fitted_params` is the model's learned parameters (eg, the coefficients in a linear
  model) in a form understood by model operations. $DOC_OPERATIONS If some training
  outcome of user-interest is not needed for operations, it should be part of `report`
  instead (see below).

- The `state` is for passing to [`LearnAPI.update!`](@ref) or
  [`LearnAPI.ingest!`](@ref). For models that implement neither, `state` should be
  `nothing`.

- The `report` records byproducts of training not in the `fitted_params`, such as feature
  rankings, or out-of-sample estimates of performance.


# New model implementations

Overloading this method for new models is optional.  A fallback performs no
computation, returning `(nothing, nothing, nothing)`.

$DOC_WHAT_IS_DATA

$DOC_MUTATING_MODELS

$(DOC_IMPLEMENTED_METHODS(:fit))

If supporting metadata, you must also implement [`LearnAPI.fit_keywords`](@ref) to list
the supported keyword argument names (e.g., `class_weights`).

See also [`LearnAPI.update!`](@ref), [`LearnAPI.ingest!`](@ref).

"""
fit(::Any, ::Any, ::Integer, data...; metadata...) = nothing, nothing, nothing


# # UPDATE

"""
    LearnAPI.update!(model, verbosity, fitted_params, state, data...; metadata...)

Based on the values of `state`, and `fitted_params` returned by a preceding call to
[`LearnAPI.fit`](ref), [`LearnAPI.ingest!`](@ref), or [`LearnAPI.update!`](@ref), update a
model's learned parameters, returning new (or mutated) `state` and `fitted_params`.

Intended for retraining a model when the training data has not changed, but `model`
properties (hyperparameters) may have changed. Specifically, the assumption is that `data`
and `metadata` have the same values seen in the most recent call to `fit/update!/ingest!`
(and will typically be ignored).

The most common use case is for continuing the training of an iterative model: `state` is
simply a copy of the model used in the last training call (`fit`, `update!` or `ingest!`)
and this will include the current number of iterations as a property. If `model` and
`state` differ only in the number of iterations (e.g., epochs in a neural network), which
has increased, then the learned parameters (weights) are updated, rather computed ab
initio. Otherwise, `update!` simply calls `fit`, to force retraining from scratch.

It is permitted to return mutated versions of `state` and `fitted_params`.

For incremental training (same model, new data) see instead [`LearnAPI.ingest!`](@ref).


# Return value

Same as [`LearnAPI.fit`](@ref), namely a tuple (`fitted_params`, `state`, `report`). See
[`LearnAPI.fit`](@ref) for details.


# New model implementations

Overloading this method is optional. A fallback calls `LearnAPIperforms.fit`:

```julia
LearnAPI.update!(model, verbosity, fitted_params, state, data...; metadata...) =
    fit(model, verbosity, data; metadata...)
```

$DOC_WHAT_IS_DATA

$DOC_MUTATING_MODELS

$(DOC_IMPLEMENTED_METHODS(:fit))

See also [`LearnAPI.fit`](@ref), [`LearnAPI.ingest!`](@ref).

"""
update!(model, verbosity, fitted_params, state, data...; metadata...) =
    fit(model, verbosity, data...; metadata...)


# # INGEST

"""
    LearnAPI.ingest!(model, verbosity, fitted_params, state, data...; metadata...)

For a model that supports incremental learning, update the learned parameters using
`data`, which has typically not been seen before.  The arguments `state` and
`fitted_params` are the output of a preceding call to [`LearnAPI.fit`](ref),
[`LearnAPI.ingest!`](@ref), or [`LearnAPI.update!`](@ref), of which mutated or new
versions are returned.

For updating learned parameters using the *same* data but new hyperparameters, see instead
[`LearnAPI.update!`](@ref).

For incremental training, see instead [`LearnAPI.ingest!`](@ref).


# Return value

Same as [`LearnAPI.fit`](@ref), namely a tuple (`fitted_params`, `state`, `report`). See
[`LearnAPI.fit`](@ref) for details.


# New model implementations

Implementing this method is an optional. It has no fallback.

$DOC_MUTATING_MODELS

$(DOC_IMPLEMENTED_METHODS(:fit))

See also [`LearnAPI.fit`](@ref), [`LearnAPI.update!`](@ref).

"""
function ingest!(model, verbosity, fitted_params, state, data...; metadata...) end
