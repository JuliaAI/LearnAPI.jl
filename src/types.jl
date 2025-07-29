# # KIND OF LEARNER

# see later for doc-string:
abstract type KindOfLearner end

"""
    LearnAPI.Standard

Type with a single instance, `LearnAPI.Standard()`.

If [`LearnAPI.kind_of(learner)`](@ref)` == LearnAPI.Standard()`, then the only possible
signatures of [`fit`](@ref), [`predict`](@ref) and [`transform`](@ref) are those appearing
below, or variations on these in which keyword arguments are also supplied:

```
model = fit(learner, data)
predict(model, new_data)
predict(model, kop::KindOfProxy, new_data)
transform(model, new_data)
```

and the one-line convenience forms

```
predict(learner, data)
predict(learner, kop::KindOfProxy, new_data)
transform(learner, data)
```

See also [`LearnAPI.Static`](@ref), [`LearnAPI.Generative`](@ref).

"""
struct Standard <: KindOfLearner end

"""
    LearnAPI.Static

Type with a single instance, `LearnAPI.Static()`.

If [`LearnAPI.kind_of(learner)`](@ref)` == LearnAPI.Static()`, then the only possible
signatures of [`fit`](@ref), [`predict`](@ref) and [`transform`](@ref) are those appearing
below, or variations on these in which keyword arguments are also supplied:

```
model = fit(learner)   # (no `data` argument)
predict(model, data)
predict(model, kop::KindOfProxy, data)
transform(model, data)
```

and the one-line convenience forms

```
predict(learner, data)
predict(learner, kop::KindOfProxy)
transform(learner, data)
```

See also [`LearnAPI.Standard](@ref), [`LearnAPI.Generative`](@ref).

"""
struct Static <: KindOfLearner end

"""
    LearnAPI.Generative

Type with a single instance, `LearnAPI.Generative()`.

If [`LearnAPI.kind_of(learner)`](@ref)` == LearnAPI.Generative()`, then the only possible
signatures of [`fit`](@ref), [`predict`](@ref) and [`transform`](@ref) are those appearing
below, or variations on these in which keyword arguments are also supplied:

```
model = fit(learner, data)
predict(model)
predict(model, kop::KindOfProxy)
transform(model)
```

and the one-liner convenience forms

```
predict(learner, data)
predict(learner, kop::KindOfProxy, data)
transform(learner, data)
```

"""
struct Generative <: KindOfLearner end


"""
    LearnAPI.KindOfLearner

Abstract type whose instances are the possible values of
[`LearnAPI.kind_of(learner)`](@ref). All instances of this type, and brief indications of
their interpretation, appear below.

[`LearnAPI.Standard()`](@ref): A typical workflow looks like:

```
model = fit(learner, data)
predict(learner, new_data)
# or
transform(learner, new_data)
```

[`LearnAPI.Static()`](@ref): A typical workflow looks like:

```
model = fit(learner)
predict(learner, data)
# or
transform(learner, data)
```

[`LearnAPI.Generative()`](@ref): A typical workflow looks like:

```
model = fit(learner, data)
predict(learner)
# or
transform(learner)
```

For precise details, refer to the document strings for [`LearnAPI.Standard`](@ref),
[`LearnAPI.Static`](@ref), and [`LearnAPI.Generative`](@ref).
"""
KindOfLearner


# # TARGET PROXIES

# see later for doc-string:
abstract type KindOfProxy end

"""
    LearnAPI.IID <: LearnAPI.KindOfProxy

Abstract subtype of [`LearnAPI.KindOfProxy`](@ref). If `kind_of_proxy` is an instance of
`LearnAPI.IID` then, given `data` consisting of ``n`` observations, the
following must hold:

- `ŷ = LearnAPI.predict(model, kind_of_proxy, data)` is
  data also consisting of ``n`` observations.

- The ``j``th observation of `ŷ`, for any ``j``, depends only on the ``j``th
  observation of the provided `data` (no correlation between observations).

Alternatively, in the case `LearnAPI.sees_features(learner) == false` (so that
`predict(model, ...)` consumes no data, and `fit` sees only target data), one requires
only that:

- `LearnAPI.predict(model, kind_of_proxy)` consists of a single observation (such as a
  single probability distribution).

See also [`LearnAPI.KindOfProxy`](@ref).

# Extended help

| type                         | form of an observation                                                                                                                                                            |
|:----------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Point`                      | same as target observations; may have the interpretation of a 50% quantile, 50% expectile or mode                                                                                 |
| `Interpolated`               | real-valued approximation/interpolation of a discrete-valued target, such as a count (e.g., number of phone calls)                                                                |
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
| `HazardFunction`             | hazard function for survival time                                                                                                                                                 |
| `OutlierScore`               | numerical score reflecting degree of outlierness (not necessarily normalized)                                                                                                     |

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

const CONCRETE_TARGET_PROXY_SYMBOLS = [
    IID_SYMBOLS...,
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

Proxy types are grouped under two abstract subtypes:

- [`LearnAPI.IID`](@ref): The main type, for proxies consisting of uncorrelated individual
  components, one for each input observation. The type also applies to learners, such as
  density estimators, that are trained on a target variable only (no features), and where
  `predict` consumes no data and the returned target proxy is a single observation (e.g.,
  a single probability mass function)

- [`LearnAPI.Joint`](@ref): For learners that predict a single probabilistic structure
  encapsulating correlations between target predictions for different input observations.

For lists of all concrete instances, refer to documentation for the relevant subtype.

"""
KindOfProxy


# # DATA INTERFACES

"""

    LearnAPI.DataInterface

Abstract supertype for singleton types designating an interface for accessing observations
within a LearnAPI.jl data object.

New learner implementations must overload [`LearnAPI.data_interface(learner)`](@ref) to
return one of the instances below if the output of [`obs`](@ref) does not implement the
default [`LearnAPI.RandomAccess()`](@ref) interface. Arrays, most tables, and all tuples
thereof, implement `RandomAccess()`.

Available instances:

- [`LearnAPI.RandomAccess()`](@ref) (default)
- [`LearnAPI.FiniteIterable()`](@ref)
- [`LearnAPI.Iterable()`](@ref)

"""
abstract type DataInterface end
abstract type Finite <: DataInterface end

"""
    LearnAPI.RandomAccess

A data interface type. We say that `data` implements the `RandomAccess` interface if
`data` implements the methods `getobs` and `numobs` from MLCore.jl. The first method
allows one to grab observations specified by an arbitrary index set, as in
`MLCore.getobs(data, [2, 3, 5])`, while the second method returns the total number of
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
`Base.getindex` and `Base.length`, which are the fallbacks for `MLCore.getobs` and
`MLCore.numobs`, and this avoids making MLCore.jl a package dependency.

See also [`LearnAPI.FiniteIterable`](@ref), [`LearnAPI.Iterable`](@ref).
"""
struct RandomAccess <: Finite end

"""
    LearnAPI.FiniteIterable

A data interface type.  We say that `data` implements the `FiniteIterable` interface if it
implements Julia's `iterate` interface, including `Base.length`, and if
`Base.IteratorSize(typeof(data)) == Base.HasLength()`. For example, this is true if `data
isa MLCore.DataLoader`, which includes the output of `MLUtils.eachobs`.

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
`MLCore.numobs` or `Base.length`.)

If [`LearnAPI.data_interface(learner)`](@ref) takes the value `Iterable()`, then
[`obs`](@ref)`(learner, ...)` is guaranteed to return objects implementing `Iterable`,
and the same holds for `obs(model, ...)`, whenever `LearnAPI.learner(model) ==
learner`.

See also [`LearnAPI.FiniteIterable`](@ref), [`LearnAPI.RandomAccess`](@ref).

"""
struct Iterable <: DataInterface end
