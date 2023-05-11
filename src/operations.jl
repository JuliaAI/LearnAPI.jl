function DOC_IMPLEMENTED_METHODS(name; overloaded=false)
    word = overloaded ? "overloaded" : "implemented"
    "If $word, you must include `:$name` in the tuple returned by the "*
    "[`LearnAPI.functions`](@ref) trait. "
end

const OPERATIONS = (:predict, :transform, :inverse_transform)
const DOC_OPERATIONS_LIST_SYMBOL = join(map(op -> "`:$op`", OPERATIONS), ", ")
const DOC_OPERATIONS_LIST_FUNCTION = join(map(op -> "`LearnAPI.$op`", OPERATIONS), ", ")

const DOC_NEW_DATA =
    "The `report` contains ancilliary byproducts of the computation, or "*
    "is `nothing`; `data` is a tuple of data objects, "*
    "generally a single object representing new observations "*
    "not seen in training. "


# # METHOD STUBS/FALLBACKS

"""
    LearnAPI.predict(algorithm, kind_of_proxy::LearnAPI.KindOfProxy, fitted_params, data...)

Return `(ŷ, report)` where `ŷ` is the predictions (a data object with target predictions
as observations) or a proxy for these, for the specified `algorithm` having learned
parameters `fitted_params` (first object returned by [`LearnAPI.fit`](@ref)`(algorithm,
...)`).  $DOC_NEW_DATA

Where available, use `kind_of_proxy=LiteralTarget()` for ordinary target predictions, and
`kind_of_proxy=Distribution()` for PDF/PMF predictions. Always available is
`kind_of_proxy=`LearnAPI.preferred_kind_of_proxy(algorithm)`.

For a full list of target proxy types, run `subtypes(LearnAPI.KindOfProxy)` and
`subtypes(LearnAPI.IID)`.

# New implementations

$(DOC_IMPLEMENTED_METHODS(:predict))

If implementing `LearnAPI.predict`, then a
[`LearnAPI.preferred_kind_of_proxy`](@ref) declaration is required, as in

```julia
LearnAPI.preferred_kind_of_proxy(::Type{<:SomeAlgorithm}) = LearnAPI.Distribution()
```

which has the shorthand

```julia
@trait SomeAlgorithm preferred_kind_of_proxy=LearnAPI.Distribution()
```

The value of this trait must be an instance `T()`, where `T <: LearnAPI.KindOfProxy`.

See also [`LearnAPI.fit`](@ref).

"""
function predict end

"""
    LearnAPI.transform(algorithm, fitted_params, data...)

Return `(output, report)`, where `output` is some kind of transformation of `data`,
provided by `algorithm`, based on the learned parameters `fitted_params` (the first object
returned by [`LearnAPI.fit`](@ref)`(algorithm, ...)`). The `fitted_params` could be
`nothing`, in the case of algorithms that do not generalize to new data. $DOC_NEW_DATA


# New implementations

$(DOC_IMPLEMENTED_METHODS(:transform))

See also [`LearnAPI.inverse_transform`](@ref), [`LearnAPI.fit`](@ref),
[`LearnAPI.predict`](@ref),

"""
function transform end

"""
    LearnAPI.inverse_transform(algorithm, fitted_params, data)

Return `(data_inverted, report)`, where `data_inverted` is valid input to the call

```julia
LearnAPI.transform(algorithm, fitted_params, data_inverted)
```
$DOC_NEW_DATA

Typically, the map

```julia
data -> first(inverse_transform(algorithm, fitted_params, data))
```

will be an inverse, approximate inverse, right inverse, or approximate right inverse, for
the map

```julia
data -> first(transform(algorithm, fitted_params, data))
```

For example, if `transform` corresponds to a projection, `inverse_transform` might be the
corresponding embedding.


# New implementations

$(DOC_IMPLEMENTED_METHODS(:transform))

See also [`LearnAPI.fit`](@ref), [`LearnAPI.predict`](@ref),

"""
function inverse_transform end

function save end
function restore end


# # TARGET PROXIES

"""

    LearnAPI.KindOfProxy

Abstract type whose concrete subtypes `T` each represent a different kind of proxy for the
target variable, associated with some algorithm. Instances `T()` are used to request the
form of target predictions in [`LearnAPI.predict`](@ref) calls.

For example, `LearnAPI.Distribution` is a concrete subtype of `LearnAPI.KindOfProxy` and
the call `LearnAPI.predict(algorithm , LearnAPI.Distribution(), data...)` returns a data
object whose observations are probability density/mass functions, assuming `algorithm`
supports predictions of that form.

Run `subtypes(LearnAPI.KindOfProxy)` and `subtypes(LearnAPI.IID)` to list all concrete
subtypes of `KindOfProxy`.

"""
abstract type KindOfProxy end

"""
    LearnAPI.IID <: LearnAPI.KindOfProxy

Abstract subtype of [`LearnAPI.KindOfProxy`](@ref). If `kind_of_proxy` is an instance of
`LearnAPI.IID` then, given `data` constisting of ``n`` observations, the following must
hold:

- `LearnAPI.predict(algorithm, kind_of_proxy, data...) == (ŷ, report)` where `ŷ` is data
  also consisting of ``n`` observations; and

- The ``j``th observation of `ŷ`, for any ``j``, depends only on the ``j``th
  observation of the provided `data` (no correlation between observations).

See also [`LearnAPI.KindOfProxy`](@ref).

"""
abstract type IID <: KindOfProxy end

struct LiteralTarget <: IID end
struct Sampleable <: IID end
struct Distribution <: IID end
struct LogDistribution <: IID end
struct Probability <: IID end
struct LogProbability <: IID end
struct Parametric <: IID end
struct LabelAmbiguous <: IID end
struct LabelAmbiguousSampleable <: IID end
struct LabelAmbiguousDistribution <: IID end
struct ConfidenceInterval <: IID end
struct Set <: IID end
struct ProbabilisticSet <: IID end
struct SurvivalFunction <: IID end
struct SurvivalDistribution <: IID end
struct OutlierScore <: IID end
struct Continuous <: IID end

struct JointSampleable <: KindOfProxy end
struct JointDistribution <: KindOfProxy end
struct JointLogDistribution <: KindOfProxy end

const CONCRETE_TARGET_PROXY_TYPES = [
    subtypes(IID)...,
    JointSampleable,
    JointDistribution,
    JointLogDistribution,
]
