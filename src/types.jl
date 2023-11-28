# # TARGET PROXIES

const DOC_HOW_TO_LIST_PROXIES =
    "Run `LearnAPI.CONCRETE_TARGET_PROXY_TYPES` "*
    " to list all options. "


"""

    LearnAPI.KindOfProxy

Abstract type whose concrete subtypes `T` each represent a different kind of proxy for
some target variable, associated with some algorithm. Instances `T()` are used to request
the form of target predictions in [`predict`](@ref) calls.

See LearnAPI.jl documentation for an explanation of "targets" and "target proxies".

For example, `Distribution` is a concrete subtype of `LearnAPI.KindOfProxy` and a call
like `predict(model, Distribution(), Xnew)` returns a data object whose observations are
probability density/mass functions, assuming `algorithm` supports predictions of that
form.

$DOC_HOW_TO_LIST_PROXIES

"""
abstract type KindOfProxy end

"""
    LearnAPI.IID <: LearnAPI.KindOfProxy

Abstract subtype of [`LearnAPI.KindOfProxy`](@ref). If `kind_of_proxy` is an instance of
`LearnAPI.IID` then, given `data` constisting of ``n`` observations, the
following must hold:

- `ŷ = LearnAPI.predict(model, kind_of_proxy, data...)` is
  data also consisting of ``n`` observations.

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

# struct None <: KindOfProxy end
struct JointSampleable <: KindOfProxy end
struct JointDistribution <: KindOfProxy end
struct JointLogDistribution <: KindOfProxy end

const CONCRETE_TARGET_PROXY_TYPES = [
    subtypes(IID)...,
    setdiff(subtypes(KindOfProxy), subtypes(IID))...,
]

const CONCRETE_TARGET_PROXY_TYPES_SYMBOLS = map(CONCRETE_TARGET_PROXY_TYPES) do T
    Symbol(last(split(string(T), '.')))
end

const CONCRETE_TARGET_PROXY_TYPES_LIST = join(
    map(CONCRETE_TARGET_PROXY_TYPES_SYMBOLS) do s
        "`$s`"
    end,
    ", ",
    " and ",
)
