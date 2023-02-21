# # DOC STRING HELPERS

const TRAINING_FUNCTIONS = (:fit, :update!, :ingest!)

const DOC_OPERATIONS =
    "An *operation* is a method, like [`LearnAPI.predict`](@ref) or "*
    "[`LearnAPI.transform`](@ref), that has signature "*
    "`(algorithm, fitted_params, data....)`; do `LearnAPI.OPERATIONS` to list."

function DOC_IMPLEMENTED_METHODS(name; overloaded=false)
    word = overloaded ? "overloaded" : "implemented"
    "If $word, you must include `:$name` in the tuple returned by the "*
    "[`LearnAPI.functions`](@ref) trait. "
end

const DOC_METADATA =
    "`metadata` is for extra information pertaining to the data that is never "*
    "iterated or subsampled, eg., weights for target classes, or feature names "*
    "(if these are not embedded in the representation of `data`). Another example "*
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

        The method is not permitted to mutate `algorithm`. In particular, if `algorithm` has a
        random number generator as a hyperparameter (property) then it must be copied
        before use.  """

# # FIT

"""
    LearnAPI.fit(algorithm, verbosity, data...; metadata...)

Perform training associated with `algorithm` using the provided `data` and `metadata`. With
the exception of warnings, training will be silent if `verbosity == 0`. Lower values
should suppress warnings; any integer ought to be admissible. Here:

- `algorithm` is a property-accessible object whose properties are the hyperparameters of
   some ML/statistical algorithm.

- `data` is a tuple of data objects with a common number of observations, for example,
  `data = (X, y, w)` where `X` is a table of features, `y` is a target vector with the
  same number of rows, and `w` a vector of per-observation weights.

- $DOC_METADATA To see the keyword names for metadata supported by `algorithm`, do
  `LearnAPI.fit_keywords(algorithm)`. "


# Return value

Returns a tuple (`fitted_params`, `state`, `report`) where:

- The `fitted_params` is the algorithm's learned parameters (eg, the coefficients in a linear
  algorithm) in a form understood by operations. $DOC_OPERATIONS If some training
  outcome of user-interest is not needed for operations, it should be part of `report`
  instead (see below).

- The `state` is for passing to [`LearnAPI.update!`](@ref) or
  [`LearnAPI.ingest!`](@ref). For algorithms that implement neither, `state` should be
  `nothing`.

- The `report` records byproducts of training not in the `fitted_params`, such as feature
  rankings, or out-of-sample estimates of performance.


# New implementations

Overloading this method for new algorithms is optional.  A fallback performs no
computation, returning `(nothing, nothing, nothing)`.

See the LearnAPI.jl documentation for the detailed requirements of LearnAPI.jl algorithm
objects.

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
    LearnAPI.update!(algorithm, verbosity, fitted_params, state, data...; metadata...)

Based on the values of `state`, and `fitted_params` returned by a preceding call to
[`LearnAPI.fit`](ref), [`LearnAPI.ingest!`](@ref), or [`LearnAPI.update!`](@ref), update a
algorithm's fitted parameters, returning new (or mutated) `state` and `fitted_params`.

Intended for retraining when the training data has not changed, but `algorithm`
properties (hyperparameters) may have changed, e.g., when increasing an iteration
parameter. Specifically, the assumption is that `data` and `metadata` have the same values
seen in the most recent call to `fit/update!/ingest!`.

For incremental training (same algorithm, new data) see instead [`LearnAPI.ingest!`](@ref).

# Return value

Same as [`LearnAPI.fit`](@ref), namely a tuple (`fitted_params`, `state`, `report`). See
[`LearnAPI.fit`](@ref) for details.


# New implementations

Overloading this method is optional. A fallback calls `LearnAPI.fit`:

```julia
LearnAPI.update!(algorithm, verbosity, fitted_params, state, data...; metadata...) =
    fit(algorithm, verbosity, data; metadata...)
```
$(DOC_IMPLEMENTED_METHODS(:fit))

$DOC_WHAT_IS_DATA

The most common use case is continuing training of an iterative algorithm: `state` is
simply a copy of the algorithm used in the last training call (`fit`, `update!` or `ingest!`)
and this will include the current number of iterations as a property. If `algorithm` and
`state` differ only in the number of iterations (e.g., epochs in a neural network), which
has increased, then the fitted parameters (weights) are updated, rather than computed from
scratch. Otherwise, `update!` simply calls `fit`, to force retraining from scratch.

It is permitted to return mutated versions of `state` and `fitted_params`.

$DOC_MUTATING_MODELS

See also [`LearnAPI.fit`](@ref), [`LearnAPI.ingest!`](@ref).

"""
update!(algorithm, verbosity, fitted_params, state, data...; metadata...) =
    fit(algorithm, verbosity, data...; metadata...)


# # INGEST

"""
    LernAPI.ingest!(algorithm, verbosity, fitted_params, state, data...; metadata...)

For an algorithm that supports incremental learning, update the fitted parameters using
`data`, which has typically not been seen before.  The arguments `state` and
`fitted_params` are the output of a preceding call to [`LearnAPI.fit`](ref),
[`LearnAPI.ingest!`](@ref), or [`LearnAPI.update!`](@ref), of which mutated or new
versions are returned.

For updating fitted parameters using the *same* data but new hyperparameters, see instead
[`LearnAPI.update!`](@ref).

For training an algorithm with new hyperparameters but *unchanged* data, see instead
[`LearnAPI.update!`](@ref).


# Return value

Same as [`LearnAPI.fit`](@ref), namely a tuple (`fitted_params`, `state`, `report`). See
[`LearnAPI.fit`](@ref) for details.


# New implementations

Implementing this method is optional. It has no fallback.

$(DOC_IMPLEMENTED_METHODS(:fit))

$DOC_MUTATING_MODELS

See also [`LearnAPI.fit`](@ref), [`LearnAPI.update!`](@ref).

"""
function ingest!(algorithm, verbosity, fitted_params, state, data...; metadata...) end
