const DOC_OPERATIONS = "An *operation* is a method like [`MLInterface.predict`](@ref) or"*
    "[`MLInterface.transform`](@ref); do `MLInterface.OPERATIONS` to list."

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

- The `state` is for passing to [`MLInterface.update`](@ref) or
  [`MLInterface.ingest`](@ref). For models that implement neither, `state` should be
  `nothing`.

- The `report` records byproducts of training not in the `fitresult`.


# Fallback

A fallback performs no computation, returning `(nothing, nothing, nothing)`.

See also [`update`](@ref), [`ingest`](@ref).

"""
fit(::Any, ::Any, ::Integer, data...; metadata...) = nothing, nothing, nothing
