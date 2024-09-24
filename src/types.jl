# # TARGET PROXIES

# see later for doc string:
abstract type KindOfProxy end

"""
    LearnAPI.IID <: LearnAPI.KindOfProxy

Abstract subtype of [`LearnAPI.KindOfProxy`](@ref). If `kind_of_proxy` is an instance of
`LearnAPI.IID` then, given `data` constisting of ``n`` observations, the
following must hold:

- `ŷ = LearnAPI.predict(model, kind_of_proxy, data)` is
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
struct LabelAmbiguousFuzzy <: IID end
struct ConfidenceInterval <: IID end
struct Fuzzy <: IID end
struct ProbabilisticFuzzy <: IID end
struct SurvivalFunction <: IID end
struct SurvivalDistribution <: IID end
struct HazardFunction <: IID end
struct OutlierScore <: IID end
struct Continuous <: IID end
struct Quantile <: IID end
struct Expectile <: IID end


"""
    Joint <: KindOfProxy

Abstract subtype of [`LearnAPI.KindOfProxy`](@ref).  If `kind_of_proxy` is an instance of
`LearnAPI.Joint` then, given `data` consisting of ``n`` observations, `predict(model,
kind_of_proxy, data)` represents a *single* probability distribution for the sample
space ``Y^n``, where ``Y`` is the space from which the target variable takes its values.

"""
abstract type Joint <: KindOfProxy end
struct JointSampleable <: Joint end
struct JointDistribution <: Joint end
struct JointLogDistribution <: Joint end

"""
    Single <: KindOfProxy

Abstract subtype of [`LearnAPI.KindOfProxy`](@ref). It applies only to algorithms for
which `predict` has no data argument, i.e., is of the form `predict(model,
kind_of_proxy)`. An example is an algorithm learning a probability distribution from
samples, and we regard the samples as drawn from the "target" variable. If in this case,
`kind_of_proxy` is an instance of `LearnAPI.Single` then, `predict(algorithm)` returns a
single object representing a probability distribution.

"""
abstract type Single <: KindOfProxy end
struct SingleSampeable <: Single end
struct SingleDistribution <: Single end
struct SingleLogDistribution <: Single end

const CONCRETE_TARGET_PROXY_TYPES = [
    subtypes(IID)...,
    subtypes(Single)...,
    subtypes(Joint)...,
]

const CONCRETE_TARGET_PROXY_TYPES_SYMBOLS = map(CONCRETE_TARGET_PROXY_TYPES) do T
    Symbol(last(split(string(T), '.')))
end

const CONCRETE_TARGET_PROXY_TYPES_LIST = join(
    map(CONCRETE_TARGET_PROXY_TYPES_SYMBOLS) do s
        "`$s()`"
    end,
    ", ",
    " and ",
)

const DOC_HOW_TO_LIST_PROXIES =
    "The instances of [`LearnAPI.KindOfProxy`](@ref) are: "*
    "$(LearnAPI.CONCRETE_TARGET_PROXY_TYPES_LIST). "


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
KindOfProxy


# # DATA INTERFACES

abstract type DataInterface end
abstract type Finite <: DataInterface end

"""
    LearnAPI.RandomAccess

A data interface type. We say that `data` implements the `RandomAccess` interface if
`data` implements the methods `getobs` and `numobs` from MLUtils.jl. The first method
allows one to grab observations specified by an arbitrary index set, as in
`MLUtils.getobs(data, [2, 3, 5])`, while the second method returns the total number of
available observations, which is assumed to be known and finite.

All arrays implement `RandomAccess`, with the last index being the observation index
(observations-as-columns in matrices).

A Tables.jl compatible table `data` implements `RandomAccess` if `Tables.istable(data)` is
true and if `data` implements `DataAPI.nrows`. This includes many tables, and in
particular, `DataFrame`s. Tables that are also tuples are excluded.

Any tuple of objects implementing `RandomAccess` also implements `RandomAccess`.

If [`LearnAPI.data_interface(algorithm)`](@ref) takes the value `RandomAccess()`, then
[`obs`](@ref)`(algorithm, ...)` is guaranteed to return objects implementing the
`RandomAccess` interface, and the same holds for `obs(model, ...)`, whenever
`LearnAPI.algorithm(model) == algorithm`.

# Implementing `RandomAccess` for new data types

Typically, to implement `RandomAccess` for a new data type requires only implementing
`Base.getindex` and `Base.length`, which are the fallbacks for `MLUtils.getobs` and
`MLUtils.numobs`, and this avoids making MLUtils.jl a package dependency.

See also [`LearnAPI.FiniteIterable`](@ref), [`LearnAPI.Iterable`](@ref).
"""
struct RandomAccess <: Finite end

"""
    LearnAPI.FiniteIterable

A data interface type.  We say that `data` implements the `FiniteIterable` interface if
it implements Julia's `iterate` interface, including `Base.length`, and if
`Base.IteratorSize(typeof(data)) == Base.HasLength()`. For example, this is true if:

- `data` implements the [`LearnAPI.RandomAccess`](@ref) interface (arrays and most tables)

- `data isa MLUtils.DataLoader`, which includes output from `MLUtils.eachobs`.

If [`LearnAPI.data_interface(algorithm)`](@ref) takes the value `FiniteIterable()`, then
[`obs`](@ref)`(algorithm, ...)` is guaranteed to return objects implementing the
`FiniteIterable` interface, and the same holds for `obs(model, ...)`, whenever
`LearnAPI.algorithm(model) == algorithm`.

See also [`LearnAPI.RandomAccess`](@ref), [`LearnAPI.Iterable`](@ref).
"""
struct FiniteIterable <: Finite end

"""
    LearnAPI.Iterable

A data interface type. We say that `data` implements the `Iterable` interface if it
implements Julia's basic `iterate` interface. (Such objects may not implement
`MLUtils.numobs` or `Base.length`.)

If [`LearnAPI.data_interface(algorithm)`](@ref) takes the value `Iterable()`, then
[`obs`](@ref)`(algorithm, ...)` is guaranteed to return objects implementing `Iterable`,
and the same holds for `obs(model, ...)`, whenever `LearnAPI.algorithm(model) ==
algorithm`.

See also [`LearnAPI.FiniteIterable`](@ref), [`LearnAPI.RandomAccess`](@ref).

"""
struct Iterable <: DataInterface end
