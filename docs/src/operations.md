# [Predict and Other Operations](@id operations)

> **Summary** An method delivering output for some algorithm which has finished learning,
> applied to (new) data, is called an **operation**.  The output depends on the fitted
> parameters associated with the algorithm, which is `nothing` for non-generalizing
> algorithms. Implement the `predict` operation when the output is predictions of a target
> variable or, more generally a proxy for the target, such as probability distributions.
> Otherwise implement `transform` and, optionally `inverse_transform`.

The methods `predict`, `transform` and `inverse_transform` are called *operations*. They
are all dispatched on an algorithm, fitted parameters and data. The `predict` operation
additionally includes a `::ProxyType` argument in position two. If [`LearnAPI.fit`](@ref)
is not implemented, then the fitted parameters will always be `nothing`.

Here's a snippet of code with a `LearnAPI.predict` call:

```julia
fitted_params, state, fit_report = LearnAPI.fit(some_algorithm, 1, X, y)
ŷ, predict_report = 
    LearnAPI.predict(some_algorithm, LearnAPI.LiteralTarget(), fitted_params, Xnew)
```

| method                             | compulsory? | fallback | requires    |
|:-----------------------------------|:-----------:|:--------:|:-----------:|
[`LearnAPI.predict`](@ref)           | no          | none     |             |
[`LearnAPI.transform`](@ref)         | no          | none     |             |
[`LearnAPI.inverse_transform`](@ref) | no          | none     | `transform` |


## General requirements

- Operations always return a tuple `(output, report)` where `output` is the usual output
  (e.g., the target predictions if the operation is `predict`) and `report`
  includes byproducts of the computation, typically `nothing` unless the algorithm does not
  generalize to new data (does not implement `fit`). 

- If implementing a `predict` method, you must also make a
  [`LearnAPI.preferred_kind_of_proxy`](@ref) declaration.
  
- The name of each operation explicitly overloaded must be included in the return value
  of the [`LearnAPI.functions`](@ref) trait.

## Predict or transform?

If the algorithm has a notion of [target variable](@ref proxy), then implement a `predict`
method for each supported kind of target proxy (`LiteralTarget()`, `Distribution()`,
etc). See [Target proxies](@ref) below.

If an operation is to have an inverse operation, then it cannot be `predict` - use
`transform`, and (optionally) `inverse_transform`, for inversion, broadly understood. See
[`LearnAPI.inverse_transform`](@ref) below.


## Target proxies

The concept of **target proxy** is defined under [Targets and target proxies](@ref
proxy). The available kinds of target proxy are classified by subtypes of
`LearnAPI.KindOfProxy`. These types are intended for dispatch only and have no fields.

```@docs
LearnAPI.KindOfProxy
```
```@docs
LearnAPI.IID
```

|          type                   | form of an observation                              | 
|:-------------------------------:|:----------------------------------------------------|
| `LearnAPI.None`                 | has no declared relationship with a target variable |
| `LearnAPI.LiteralTarget`           | same as target observations |
| `LearnAPI.Sampleable`           | object that can be sampled to obtain object of the same form as target observation |
| `LearnAPI.Distribution`         | explicit probability density/mass function whose sample space is all possible target observations |
| `LearnAPI.LogDistribution`      | explicit log-probability density/mass function whose sample space is possible target observations |
|  † `LearnAPI.Probability`       | raw numerical probability or probability vector |
|  † `LearnAPI.LogProbability`    | log-probability or log-probability vector | 
|  † `LearnAPI.Parametric`        | a list of parameters (e.g., mean and variance) describing some distribution |
| `LearnAPI.LabelAmbiguous`       | collections of labels (in case of multi-class target) but without a known correspondence to the original target labels (and of possibly different number) as in, e.g., clustering | 
| `LearnAPI.LabelAmbiguousSampleable`  | sampleable version of `LabelAmbiguous`; see `Sampleable` above  |
| `LearnAPI.LabelAmbiguousDistribution`| pdf/pmf version of `LabelAmbiguous`; see `Distribution`  above  |
| `LearnAPI.ConfidenceInterval`   | confidence interval (possible requirement:  observation `isa Tuple{Real,Real}`) |
| `LearnAPI.Set`                  | finite but possibly varying number of target observations |
| `LearnAPI.ProbabilisticSet`     | as for `Set` but labeled with probabilities (not necessarily summing to one) |
| `LearnAPI.SurvivalFunction`     | survival function (possible requirement: observation is single-argument function mapping `Real` to `Real`) |
| `LearnAPI.SurvivalDistribution` | probability distribution for survival time |
| `LearnAPI.OutlierScore`         | numerical score reflecting degree of outlierness (not necessarily normalized) |
| `LearnAPI.Continuous`           | real-valued approximation/interpolation of a discrete-valued target, such as a count (e.g., number of phone calls) |

† Provided for completeness but discouraged to avoid [ambiguities in
representation](https://github.com/alan-turing-institute/MLJ.jl/blob/dev/paper/paper.md#a-unified-approach-to-probabilistic-predictions-and-their-evaluation).

> Table of concrete subtypes of `LearnAPI.IID <: LearnAPI.KindOfProxy`.

In the following table of subtypes `T <: LearnAPI.KindOfProxy` not falling under the `IID`
umbrella, the first return value of `predict(algorithm, ::T, fitted_params, data...)` is
not divided into individual observations, but represents a *single* probability
distribution for the sample space ``Y^n``, where ``Y`` is the space the target variable
takes its values, and `n` is the number of observations in `data`.

|          type `T`               | form of output of `predict(algorithm, ::T, fitted_params, data...)` |
|:-------------------------------:|:--------------------------------------------------------------------------|
| `LearnAPI.JointSampleable`      | object that can be sampled to obtain a *vector* whose elements have the form of target observations; the vector length matches the number of observations in `data`. |
| `LearnAPI.JointDistribution`    | explicit probability density/mass function whose sample space is vectors of target observations;  the vector length matches the number of observations in `data` |
| `LearnAPI.JointLogDistribution` | explicit log-probability density/mass function whose sample space is vectors of target observations;  the vector length matches the number of observations in `data` |

> Table of `LearnAPI.KindOfProxy` subtypes not subtyping `LearnAPI.IID`


## Reference

```@docs
LearnAPI.predict
LearnAPI.transform
LearnAPI.inverse_transform
```
