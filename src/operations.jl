const PREDICT_OPERATIONS = (:predict,
                            :predict_mode,
                            :predict_mean,
                            :predict_median,
                            :predict_joint)
const OPERATIONS = (PREDICT_OPERATIONS..., :transform, :inverse_transform)

const DOC_NEW_DATA =
    "Here `report` contains ancilliary byproducts of the computation, or "*
    "is `nothing`; `data` is a tuple of data objects, "*
    "generally a single object representing new observations "*
    "not seen in training. "


# # METHOD STUBS/FALLBACKS

"""
    LearnAPI.predict(model, fitted_params, data...)

Return `(ŷ, report)` where `ŷ` are the predictions, or prediction-like output, for
a machine learning model, `model`, with learned parameters `fitted_params`, as returned by
[`LearnAPI.fit`](@ref).  $DOC_NEW_DATA


# New model implementations

$(DOC_IMPLEMENTED_METHODS(:predict))

If `performance_measureable = true`, then `ŷ` must be:

- either an array or table with the same number of observations as each element of `data`;
  it cannot be a lazy object, such as a `DataLoader`

- **target-like**; see  [`LearnAPI.paradigm`](@ref) for specifics.

Otherwise there are no restrictions on what `predict` may return, apart from what the
implementation itself promises, by making an optional [`LearnAPI.output_scitypes`](@ref)
declaration.

By default, it is expected that `data` has length one. Otherwise,
[`LearnAPI.input_scitypes`](@ref) must be overloaded.

See also [`LearnAPI.fit`](@ref), [`LearnAPI.predict_mean`](@ref),
[`LearnAPI.predict_mode`](@ref), [`LearnAPI.predict_median`](@ref).

"""
function predict end

function DOC_PREDICT(reducer)
    operation = Symbol(string("predict_", reducer))
    extra = DOC_IMPLEMENTED_METHODS(operation, overloaded=true)
    """
        LearnAPI.predict_$reducer(model, fitted_params, data...)

    Same as [`LearnAPI.predict`](@ref) except replace probababilistic predictions with
    $reducer values.

    # New model implementations

    A fallback broadcasts `$reducer` over the first return value `ŷ` of
    `LearnAPI.predict`. An algorithm that computes probabilistic predictions may already
    need to predict mean values, and so overloading this method might enable a performance
    boost.

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
            distributions, report = predict(args...)
            yhat = $reducer.(distributions)
            return (yhat, report)
        end
    end |> eval
end

"""
    LearnAPI.predict_joint(model, fitted_params, data...)

For a supervised learning model, return `(d, report)`, where `d` is a *single* probability
distribution for the sample space ``Y^n``, whose elements are `n`-dimensional vectors with
element type matching that of the training target (the second data object in
`LearnAPI.fit(model, verbosity, data...)`). Here `n` is the number of observations in
`data`.  Here `fitted_params` are the model's learned parameters, as returned by
[`LearnAPI.fit`](@ref). $DOC_NEW_DATA.

While the interpretation of this distribution depends on the model, marginalizing
component-wise will generally deliver *correlated* univariate distributions, and these will
generally not agree with those returned by `LearnAPI.predict`, if also implemented.

# New model implementations

It is not necessary that `LearnAPI.predict` be implemented but
`LearnAPI.performance_measureable(model)` must return `true`.

$(DOC_IMPLEMENTED_METHODS(:predict_joint)).

See also [`LearnAPI.fit`](@ref), [`LearnAPI.predict`](@ref).

"""
function predict_joint end

"""
    LearnAPI.transform(model, fitted_params, data...)

Return `(output, report)`, where `output` is some kind of transformation of `data`, provided
by `model`, based on the learned parameters `fitted_params`, as returned by
[`LearnAPI.fit`](@ref) (which could be `nothing` for models that do not generalize to new
data, such as "static transformers"). $DOC_NEW_DATA


# New model implementations

$(DOC_IMPLEMENTED_METHODS(:transform))

By default, it is expected that `data` has length one. Otherwise,
[`LearnAPI.input_scitypes`](@ref) must be overloaded.

See also [`LearnAPI.fit`](@ref), [`MJInterface.predict`](@ref),

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
