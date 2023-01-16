const OPERATIONS = (:predict, :predict_joint, :transform, :inverse_transform)
const DOC_OPERATIONS_LIST = join(map(op -> "`:$op`", OPERATIONS), ", ")

const DOC_NEW_DATA =
    "The `report` contains ancilliary byproducts of the computation, or "*
    "is `nothing`; `data` is a tuple of data objects, "*
    "generally a single object representing new observations "*
    "not seen in training. "


# # METHOD STUBS/FALLBACKS

"""
    LearnAPI.predict(model, fitted_params, data...)

Return `(ŷ, report)` where `ŷ` are the predictions, or prediction-like output (such as
probabilities), for a machine learning model `model`, with learned parameters
`fitted_params` (first object returned by [`LearnAPI.fit`](@ref)`(model, ...)`).
$DOC_NEW_DATA

# New model implementations

$(DOC_IMPLEMENTED_METHODS(:predict))

If `predict` is computing a target proxy, as defined in the LearnAPI documentation, then a
[`LearnAPI.predict_proxy`](@ref) declaration is required, as in

```julia
LearnAPI.predict_proxy(::Type{<:SomeModel}) = LearnAPI.Distribution()
```

which has the shorthand

```julia
@trait SomeModel predict_proxy=LearnAPI.Distribution()
```

The value of this trait must be an instance `T()`, where `T <: LearnAPI.TargetProxy`.

See also [`LearnAPI.fit`](@ref).

"""
function predict end

# function DOC_PREDICT(reducer)
#     operation = Symbol(string("predict_", reducer))
#     extra = DOC_IMPLEMENTED_METHODS(operation, overloaded=true)
#     """
#         LearnAPI.predict_$reducer(model, fitted_params, data...)

#     Same as [`LearnAPI.predict`](@ref) except replaces probababilistic predictions with
#     $reducer values.

#     # New model implementations

#     A fallback broadcasts `$reducer` over the first return value `ŷ` of
#     `LearnAPI.predict`. An algorithm that computes probabilistic predictions may already
#     need to predict mean values, and so overloading this method might enable a performance
#     boost.

#     $extra

#     See also [`LearnAPI.predict`](@ref), [`LearnAPI.fit`](@ref).

#     """
# end

# for reducer in [:mean, :median]
#     operation = Symbol(string("predict_", reducer))
#     docstring = DOC_PREDICT(reducer)
#     quote
#         "$($docstring)"
#         function $operation(args...)
#             distributions, report = predict(args...)
#             yhat = $reducer.(distributions)
#             return (yhat, report)
#         end
#     end |> eval
# end

"""
    LearnAPI.predict_joint(model, fitted_params, data...)

For a supervised learning model, return `(d, report)`, where `d` is some representation of
the *single* probability distribution for the sample space ``Y^n``. Here ``Y`` is the
space in which the target variable associated with `model` takes its values, and `n` is
the number of observations in `data`. The specific form of the representation is given by
[`LearnAPI.predict_joint_proxy(model)`](@ref).

Here `fitted_params` are the model's learned parameters (the first object returned by
[`LearnAPI.fit`](@ref)). $DOC_NEW_DATA.

While the interpretation of this distribution depends on the model, marginalizing
component-wise will generally deliver `n` *correlated* distributions, and these will
generally not agree with those returned by `LearnAPI.predict` on the same the same `n`
input observations, if also implemented.

# New model implementations

Only implement this method if `model` has an associated concept of target variable, as
defined in the LearnAPI.jl documentation. A trait declaration for
[`LearnAPI.predict_joint_proxy`](@ref) is required, such as

```julia
LearnAPI.predict_joint_proxy(::Type{SomeModel}) = JointSampleable()
```

which has the shorhand

```julia
@trait SomeModel predict_joint_proxy=JointSampleable()
```

The possible values for this trait are: `LearnAPI.JointSampleable()`,
`LearnAPI.JointDistribution`, and `LearnAPI.JointLogDistribution()`.

$(DOC_IMPLEMENTED_METHODS(:predict_joint)).

See also [`LearnAPI.fit`](@ref), [`LearnAPI.predict`](@ref).

"""
function predict_joint end

"""
    LearnAPI.transform(model, fitted_params, data...)

Return `(output, report)`, where `output` is some kind of transformation of `data`,
provided by `model`, based on the learned parameters `fitted_params` (the first object
returned by [`LearnAPI.fit`](@ref)`(model, ...)`). The `fitted_params` could be `nothing`,
in the case of models that do not generalize to new data. $DOC_NEW_DATA


# New model implementations

$(DOC_IMPLEMENTED_METHODS(:transform))

See also [`LearnAPI.inverse_transform`](@ref), [`LearnAPI.fit`](@ref),
[`LearnAPI.predict`](@ref),

"""
function transform end

"""
    LearnAPI.inverse_transform(model, fitted_params, data)

Return `(data_inverted, report)`, where `data_inverted` is valid input to the call

```julia
LearnAPI.transform(model, fitted_params, data_inverted)
```
$DOC_NEW_DATA

Typically, the map

```julia
data -> first(inverse_transform(model, fitted_params, data))
```

will be an inverse, approximate inverse, right inverse, or approximate right inverse, for
the map

```julia
data -> first(transform(model, fitted_params, data))
```

For example, if `transform` corresponds to a projection, `inverse_transform` might be the
corresponding embedding.


# New model implementations

$(DOC_IMPLEMENTED_METHODS(:transform))

See also [`LearnAPI.fit`](@ref), [`LearnAPI.predict`](@ref),

"""
function inverse_transform end

function save end
function restore end


# # TARGET PROXIES

abstract type TargetProxy end

struct None <: TargetProxy end 
struct TrueTarget <: TargetProxy end
struct Sampleable <: TargetProxy end
struct Distribution <: TargetProxy end
struct LogDistribution <: TargetProxy end
struct Probability <: TargetProxy end
struct LogProbability <: TargetProxy end
struct Parametric <: TargetProxy end
struct LabelAmbiguous <: TargetProxy end
struct LabelAmbiguousSampleable <: TargetProxy end
struct LabelAmbiguousDistribution <: TargetProxy end
struct ConfidenceInterval <: TargetProxy end
struct Set <: TargetProxy end
struct ProbabilisticSet <: TargetProxy end
struct SurvivalFunction <: TargetProxy end
struct SurvivalDistribution <: TargetProxy end

struct JointSampleable <: TargetProxy end
struct JointDistribution <: TargetProxy end
struct JointLogDistribution <: TargetProxy end
