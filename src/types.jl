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

# Extended help

| type                                  | form of an observation                                                                                                                                                            |
|:-------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Point`              | same as target observations; may have the interpretation of a 50% quantile, 50% expectile or mode                                                                                 |
| `Sampleable`                 | object that can be sampled to obtain object of the same form as target observation                                                                                                |
| `Distribution`               | explicit probability density/mass function whose sample space is all possible target observations                                                                                 |
| `LogDistribution`            | explicit log-probability density/mass function whose sample space is possible target observations                                                                                 |
| `Probability`¹               | numerical probability or probability vector                                                                                                                                       |
| `LogProbability`¹            | log-probability or log-probability vector                                                                                                                                         |
| `Parametric`¹                | a list of parameters (e.g., mean and variance) describing some distribution                                                                                                       |
| `LabelAmbiguous`             | collections of labels (in case of multi-class target) but without a known correspondence to the original target labels (and of possibly different number) as in, e.g., clustering |
| `LabelAmbiguousSampleable`   | sampleable version of `LabelAmbiguous`; see `Sampleable` above                                                                                                                    |
| `LabelAmbiguousDistribution` | pdf/pmf version of `LabelAmbiguous`; see `Distribution`  above                                                                                                                    |
| `LabelAmbiguousFuzzy`        | same as `LabelAmbiguous` but with multiple values of indeterminant number                                                                                                         |
| `Quantile`²                  | same as target but with quantile interpretation                                                                                                                                   |
| `Expectile`²                 | same as target but with expectile interpretation                                                                                                                                  |
| `ConfidenceInterval`²        | confidence interval                                                                                                                                                               |
| `Fuzzy`                      | finite but possibly varying number of target observations                                                                                                                         |
| `ProbabilisticFuzzy`         | as for `Fuzzy` but labeled with probabilities (not necessarily summing to one)                                                                                                    |
| `SurvivalFunction`           | survival function                                                                                                                                                                 |
| `SurvivalDistribution`       | probability distribution for survival time                                                                                                                                        |
| `SurvivalHazardFunction`     | hazard function for survival time                                                                                                                                                 |
| `OutlierScore`               | numerical score reflecting degree of outlierness (not necessarily normalized)                                                                                                     |
| `Interpolated`               | real-valued approximation/interpolation of a discrete-valued target, such as a count (e.g., number of phone calls)                                                                |

¹Provided for completeness but discouraged to avoid [ambiguities in
representation](https://github.com/alan-turing-institute/MLJ.jl/blob/dev/paper/paper.md#a-unified-approach-to-probabilistic-predictions-and-their-evaluation).

²The level will be controlled by a hyper-parameter; models providing only quantiles or
expectiles at 50% will provide `Point` instead.

"""
abstract type IID <: KindOfProxy end

const IID_SYMBOLS = [
    :Point,
    :Sampleable,
    :Distribution,
    :LogDistribution,
    :Probability,
    :LogProbability,
    :Parametric,
    :LabelAmbiguous,
    :LabelAmbiguousSampleable,
    :LabelAmbiguousDistribution,
    :LabelAmbiguousFuzzy,
    :ConfidenceInterval,
    :Fuzzy,
    :ProbabilisticFuzzy,
    :SurvivalFunction,
    :SurvivalDistribution,
    :HazardFunction,
    :OutlierScore,
    :Interpolated,
    :Quantile,
    :Expectile,
]

for S in IID_SYMBOLS
    quote
        struct $S <: IID end
    end |> eval
end


"""
    Joint <: KindOfProxy

Abstract subtype of [`LearnAPI.KindOfProxy`](@ref).  If `kind_of_proxy` is an instance of
`LearnAPI.Joint` then, given `data` consisting of ``n`` observations, `predict(model,
kind_of_proxy, data)` represents a *single* probability distribution for the sample
space ``Y^n``, where ``Y`` is the space from which the target variable takes its values.

| type `T`                        | form of output of `predict(model, ::T, data)`                                                                                                                     |
|:-------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `JointSampleable`      | object that can be sampled to obtain a *vector* whose elements have the form of target observations; the vector length matches the number of observations in `data`. |
| `JointDistribution`    | explicit probability density/mass function whose sample space is vectors of target observations;  the vector length matches the number of observations in `data`     |
| `JointLogDistribution` | explicit log-probability density/mass function whose sample space is vectors of target observations;  the vector length matches the number of observations in `data` |

"""
abstract type Joint <: KindOfProxy end

const JOINT_SYMBOLS = [
    :JointSampleable,
    :JointDistribution,
    :JointLogDistribution,
]

for S in JOINT_SYMBOLS
    quote
        struct $S <: Joint end
    end |> eval
end

"""
    Single <: KindOfProxy

Abstract subtype of [`LearnAPI.KindOfProxy`](@ref). It applies only to learners for
which `predict` has no data argument, i.e., is of the form `predict(model,
kind_of_proxy)`. An example is an algorithm learning a probability distribution from
samples, and we regard the samples as drawn from the "target" variable. If in this case,
`kind_of_proxy` is an instance of `LearnAPI.Single` then, `predict(learner)` returns a
single object representing a probability distribution.

| type `T`                         | form of output of `predict(model, ::T)`                                |
|:--------------------------------:|:-----------------------------------------------------------------------|
| `SingleSampleable`      | object that can be sampled to obtain a single target observation       |
| `SingleDistribution`    | explicit probability density/mass function for sampling the target     |
| `SingleLogDistribution` | explicit log-probability density/mass function for sampling the target |

"""
abstract type Single <: KindOfProxy end

const SINGLE_SYMBOLS = [
    :SingleSampeable,
    :SingleDistribution,
    :SingleLogDistribution,
]

for S in SINGLE_SYMBOLS
    quote
        struct $S <: Single end
    end |> eval
end

const CONCRETE_TARGET_PROXY_SYMBOLS = [
    IID_SYMBOLS...,
    SINGLE_SYMBOLS...,
    JOINT_SYMBOLS...,
]

"""

    LearnAPI.KindOfProxy

Abstract type whose concrete subtypes `T` each represent a different kind of proxy for
some target variable, associated with some learner. Instances `T()` are used to request
the form of target predictions in [`predict`](@ref) calls.

See LearnAPI.jl documentation for an explanation of "targets" and "target proxies".

For example, `Distribution` is a concrete subtype of `IID <: LearnAPI.KindOfProxy` and a
call like `predict(model, Distribution(), Xnew)` returns a data object whose observations
are probability density/mass functions, assuming `learner = LearnAPI.learner(model)`
supports predictions of that form, which is true if `Distribution() in`
[`LearnAPI.kinds_of_proxy(learner)`](@ref).

Proxy types are grouped under three abstract subtypes:

- [`LearnAPI.IID`](@ref): The main type, for proxies consisting of uncorrelated individual
  components, one for each input observation

- [`LearnAPI.Joint`](@ref): For learners that predict a single probabilistic structure
  encapsulating correlations between target predictions for different input observations

- [`LearnAPI.Single`](@ref): For learners, such as density estimators, that are trained on
  a target variable only (no features); `predict` consumes no data and the returned target
  proxy is a single probabilistic structure.

For lists of all concrete instances, refer to documentation for the relevant subtype.

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
true and if `data` implements `DataAPI.nrow`. This includes many tables, and in
particular, `DataFrame`s. Tables that are also tuples are explicitly excluded.

Any tuple of objects implementing `RandomAccess` also implements `RandomAccess`.

If [`LearnAPI.data_interface(learner)`](@ref) takes the value `RandomAccess()`, then
[`obs`](@ref)`(learner, ...)` is guaranteed to return objects implementing the
`RandomAccess` interface, and the same holds for `obs(model, ...)`, whenever
`LearnAPI.learner(model) == learner`.

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

- `data` implements the [`LearnAPI.RandomAccess`](@ref) interface (arrays and most
  tables); or

- `data isa MLUtils.DataLoader`, which includes output from `MLUtils.eachobs`.

If [`LearnAPI.data_interface(learner)`](@ref) takes the value `FiniteIterable()`, then
[`obs`](@ref)`(learner, ...)` is guaranteed to return objects implementing the
`FiniteIterable` interface, and the same holds for `obs(model, ...)`, whenever
`LearnAPI.learner(model) == learner`.

See also [`LearnAPI.RandomAccess`](@ref), [`LearnAPI.Iterable`](@ref).
"""
struct FiniteIterable <: Finite end

"""
    LearnAPI.Iterable

A data interface type. We say that `data` implements the `Iterable` interface if it
implements Julia's basic `iterate` interface. (Such objects may not implement
`MLUtils.numobs` or `Base.length`.)

If [`LearnAPI.data_interface(learner)`](@ref) takes the value `Iterable()`, then
[`obs`](@ref)`(learner, ...)` is guaranteed to return objects implementing `Iterable`,
and the same holds for `obs(model, ...)`, whenever `LearnAPI.learner(model) ==
learner`.

See also [`LearnAPI.FiniteIterable`](@ref), [`LearnAPI.RandomAccess`](@ref).

"""
struct Iterable <: DataInterface end
