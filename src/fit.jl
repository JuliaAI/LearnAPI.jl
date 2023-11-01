# # DOC STRING HELPERS

const TRAINING_FUNCTIONS = (:fit,)


# # FIT

"""
    LearnAPI.fit(algorithm, data...; verbosity=1)

Execute the algorithm with configuration `algorithm` using the provided training `data`,
returning an object, `model`, on which other methods, such as [`predict`](@ref) or
[`transform`](@ref), can be dispatched.  [`LearnAPI.functions(algorithm)`](@ref) returns a
list of methods that can be applied to either `algorithm` or `model`.

# Arguments

- `algorithm`: property-accessible object whose properties are the hyperparameters of
   some ML/statistical algorithm

$(DOC_ARGUMENTS(:fit))

- `verbosity=1`: logging level; set to `0` for warnings only, and `-1` for silent training

See also [`obsfit`](@ref), [`predict`](@ref), [`transform`](@ref),
[`inverse_transform`](@ref), [`LearnAPI.functions`](@ref), [`obs`](@ref).

# Extended help

# New implementations

LearnAPI.jl provides the following defintion of `fit`, which is never directly overloaded:

```julia
fit(algorithm, data...; verbosity=1) =
    obsfit(algorithm, Obs(), obs(fit, algorithm, data...); verbosity)
```

Rather, new algorithms should overload [`obsfit`](@ref). See also [`obs`](@ref).

"""
fit(algorithm, data...; verbosity=1) =
    obsfit(algorithm, obs(fit, algorithm, data...), verbosity)

"""
    obsfit(algorithm, obsdata; verbosity=1)

A lower-level alternative to [`fit`](@ref), this method consumes a pre-processed form of
user data.  Specifically, the following two code snippets are equivalent:

```julia
model = fit(algorithm, data...)
```
and

```julia
obsdata = obs(fit, algorithm, data...)
model = obsfit(algorithm, obsdata)
```

Here `obsdata` is algorithm-specific, "observaton-accessible" data, meaning it implements
the MLUtils.jl `getobs`/`numobs` interface for observation resampling (even if `data` does
not). Morevoer, resampled versions of `obsdata` may be passed to `obsfit` in its place.

The use of `obsfit` may offer performance advantages.  See more at [`obs`](@ref).

See also [`fit`](@ref), [`obs`](@ref).

# Extended help

# New implementations

Implementation of the following method signature is compulsory for all new algorithms:

```julia
LearnAPI.obsfit(algorithm, obsdata, verbosity)
```

Here `obsdata` has the form explained above. If [`obs`](@ref)`(fit, ...)` is not being
overloaded, then a fallback gives `obsdata = data` (always a tuple!). Note that
`verbosity` is a positional argument, not a keyword argument in the overloaded signature.

New implementations must also implement [`LearnAPI.algorithm`](@ref).

If overloaded, then the functions `LearnAPI.obsfit` and `LeranAPI.fit` must be included in
the tuple returned by the [`LearnAPI.functions(algorithm)`](@ref) trait.

## Non-generalizing algorithms

If the algorithm does not generalize to new data (e.g, DBSCAN clustering) then `data = ()`
and `obsfit` carries out no computation, as this happen entirely in a `transform` and/or
`predict` call. In such cases, `obsfit(algorithm, ...)` may return `algorithm`, but
another possibility is allowed: To provide a mechanism for `transform`/`predict` to report
byproducts of the computation (e.g., a list of boundary points in DBSCAN clustering) they
are allowed to *mutate* the `model` object returned by `obsfit`, which is then arranged to
be a mutable struct wrapping `algorithm` and fields to store the byproducts. In that case,
[`LearnAPI.predict_or_transform_mutates(algorithm)`](@ref) must be overloaded to return
`true`.

"""
obsfit(algorithm, obsdata; verbosity=1) =
    obsfit(algorithm, obsdata, verbosity)


# # UPDATE

"""
    LearnAPI.update!(algorithm, verbosity, fitted_params, state, data...)

Based on the values of `state`, and `fitted_params` returned by a preceding call to
[`LearnAPI.fit`](@ref), [`LearnAPI.ingest!`](@ref), or [`LearnAPI.update!`](@ref), update a
algorithm's fitted parameters, returning new (or mutated) `state` and `fitted_params`.

Intended for retraining when the training data has not changed, but `algorithm`
properties (hyperparameters) may have changed, e.g., when increasing an iteration
parameter. Specifically, the assumption is that `data` have the same values
seen in the most recent call to `fit/update!/ingest!`.

For incremental training (same algorithm, new data) see instead [`LearnAPI.ingest!`](@ref).

# Return value

Same as [`LearnAPI.fit`](@ref), namely a tuple (`fitted_params`, `state`, `report`). See
[`LearnAPI.fit`](@ref) for details.


# New implementations

Overloading this method is optional. A fallback calls `LearnAPI.fit`:

```julia
LearnAPI.update!(algorithm, verbosity, fitted_params, state, data...) =
    fit(algorithm, verbosity, data)
```
$(DOC_IMPLEMENTED_METHODS(:fit))

The most common use case is continuing training of an iterative algorithm: `state` is
simply a copy of the algorithm used in the last training call (`fit`, `update!` or
`ingest!`) and this will include the current number of iterations as a property. If
`algorithm` and `state` differ only in the number of iterations (e.g., epochs in a neural
network), which has increased, then the fitted parameters (network weights and biases) are
updated, rather than computed from scratch. Otherwise, `update!` simply calls `fit`, to
force retraining from scratch.

It is permitted to return mutated versions of `state` and `fitted_params`.

See also [`LearnAPI.fit`](@ref), [`LearnAPI.ingest!`](@ref).

"""


# # INGEST

"""
    LernAPI.ingest!(algorithm, verbosity, fitted_params, state, data...)

For an algorithm that supports incremental learning, update the fitted parameters using
`data`, which has typically not been seen before.  The arguments `state` and
`fitted_params` are the output of a preceding call to [`LearnAPI.fit`](@ref),
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

See also [`LearnAPI.fit`](@ref), [`LearnAPI.update!`](@ref).

"""
