const PREDICT_OPERATIONS = (:predict,
                            :predict_mode,
                            :predict_mean,
                            :predict_median,
                            :predict_joint)
const OPERATIONS = (PREDICT_OPERATIONS..., :transform, :inverse_transform)

const DOC_NEW_DATA =
    "Here `data` is a tuple of data objects, "*
    "generally a single object representing new observations "*
    "not seen in training. "

# # FALLBACK HELPERS

function _predict(model, args...)
    p = predict(model, args...)
    if :predict in reporting_operations(model)
        return p
    else
        return (p, nothing)
    end
end

_compress(yhat, report) = (yhat, report)
_compress(yhat, ::Nothing) = yhat


# # METHOD STUBS/FALLBACKS

"""
    LearnAPI.predict(model, fitted_params, data...)

Return predictions or prediction-like output, `ŷ`, for a machine learning model, `model`,
with learned parameters `fitted_params`, as returned by
[`LearnAPI.fit`](@ref). $DOC_NEW_DATA

However, in the special case that `:predict in LearnAPI.reporting_operations(model)` is
`true`, `(ŷ, report)` is returned instead. Here `report` contains ancilliary byproducts of
computing the prediction.


# New model implementations

$(DOC_IMPLEMENTED_METHODS(:predict))

If `is_supervised(model) = true`, then `ŷ` must be:

- either an array or table with the same number of observations as each element of `data`;
  it cannot be a lazy object, such as a `DataLoader`

- **target-like** (point, probabilistic, or interval); see
  [`LearnAPI.prediction_type`](@ref) for specifics.

Otherwise there are no restrictions on what `predict` may return, apart from what the
implementation itself promises, by making an optional [`LearnAPI.output_scitypes`](@ref)
declaration.

By default, it is expected that `data` has length one. Otherwise,
[`LearnAPI.input_scitypes`](@ref) must be overloaded.

See also [`LearnAPI.fit`](@ref), [`MJInterface.predict_mean`](@ref),
[`LearnAPI.predict_mode`](@ref), [`LearnAPI.predict_median`](@ref).

"""
function predict end

function DOC_PREDICT(reducer)
    operation = Symbol(string("predict_", reducer))
    extra = DOC_IMPLEMENTED_METHODS(operation, overloaded=true)
    """
        LearnAPI.predict_$reducer(model, fitted_params, data...)

    If `LearnAPI.predict` returns a vector of probabilistic predictions, `distributions`,
    return a corresponding data object `ŷ` of $reducer values. $DOC_NEW_DATA

    In the special case that `LearnAPI.predict` instead returns `(distributions,
    report)`, `$operation` instead return `(ŷ, report)`.


    # New model implementations

    A fallback broadcasts `$reducer` over `ŷ`. An algorithm that predicts probabilistic
    predictions may already need to predict mean values, and so overloading this method
    might provide a performance advantage.

    $extra

    See also [`LearnAPI.predict`](@ref), [`LearnAPI.fit`](@ref).

    """
end

for reducer in [:mean, :median]
    operation = Symbol(string("predict_", reducer))
    docstring = DOC_PREDICT(reducer)
    quote
        "$($docstring)"
        function $operation(args...)
            distributions, report = _predict(args...)
            yhat = $reducer.(distributions)
            return _compress(yhat, report)
        end
    end |> eval
end

"""
    LearnAPI.predict_joint(model, fitted_params, data...)

For a supervised learning model, return a single probability distribution for the sample
space ``Y^n``, whose elements are `n`-dimensional vectors with element type matching that of
the training target (the second data object in `LearnAPI.fit(model, verbosity,
data...)`). Here `n` is the number of observations in `data`.  Here `fitted_params` are the
model's learned parameters, as returned by [`LearnAPI.fit`](@ref). $DOC_NEW_DATA.

While the interpretation of this distribution depends on the model, marginalizing
component-wise will generally deliver *correlated* univariate distributions, and these will
generally not agree with those returned by `LearnAPI.predict`, if implemented.

# New model implementations

It is not necessary that `LearnAPI.predict` be implemented but
`LearnAPI.is_supervised(model)` must return `true`.

$(DOC_IMPLEMENTED_METHODS(:predict_joint)).

See also [`LearnAPI.fit`](@ref), [`LearnAPI.predict`](@ref).

"""
function predict_joint end

"""
`Unsupervised` models must implement the `transform` operation.
"""
function transform end

"""

`Unsupervised` models may implement the `inverse_transform` operation.

"""
function inverse_transform end

# models can optionally overload these for enable serialization in a
# custom format:
function save end
function restore end
