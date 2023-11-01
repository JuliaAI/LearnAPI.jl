# [Kinds of Target Proxy](@id proxy_types)

The available kinds of [target proxy](@ref proxy) are classified by subtypes of
`LearnAPI.KindOfProxy`. These types are intended for dispatch only and have no fields.

```@docs
LearnAPI.KindOfProxy
```
```@docs
LearnAPI.IID
```

## Simple target proxies (subtypes of `LearnAPI.IID`)

| type                                  | form of an observation                                                                                                                                                            |
|:-------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LearnAPI.LiteralTarget`              | same as target observations                                                                                                                                                       |
| `LearnAPI.Sampleable`                 | object that can be sampled to obtain object of the same form as target observation                                                                                                |
| `LearnAPI.Distribution`               | explicit probability density/mass function whose sample space is all possible target observations                                                                                 |
| `LearnAPI.LogDistribution`            | explicit log-probability density/mass function whose sample space is possible target observations                                                                                 |
| † `LearnAPI.Probability`              | numerical probability or probability vector                                                                                                                                       |
| † `LearnAPI.LogProbability`           | log-probability or log-probability vector                                                                                                                                         |
| † `LearnAPI.Parametric`               | a list of parameters (e.g., mean and variance) describing some distribution                                                                                                       |
| `LearnAPI.LabelAmbiguous`             | collections of labels (in case of multi-class target) but without a known correspondence to the original target labels (and of possibly different number) as in, e.g., clustering |
| `LearnAPI.LabelAmbiguousSampleable`   | sampleable version of `LabelAmbiguous`; see `Sampleable` above                                                                                                                    |
| `LearnAPI.LabelAmbiguousDistribution` | pdf/pmf version of `LabelAmbiguous`; see `Distribution`  above                                                                                                                    |
| `LearnAPI.ConfidenceInterval`         | confidence interval                                                                                                                                                               |
| `LearnAPI.Set`                        | finite but possibly varying number of target observations                                                                                                                         |
| `LearnAPI.ProbabilisticSet`           | as for `Set` but labeled with probabilities (not necessarily summing to one)                                                                                                      |
| `LearnAPI.SurvivalFunction`           | survival function                                                                                                                                                                 |
| `LearnAPI.SurvivalDistribution`       | probability distribution for survival time                                                                                                                                        |
| `LearnAPI.OutlierScore`               | numerical score reflecting degree of outlierness (not necessarily normalized)                                                                                                     |
| `LearnAPI.Continuous`                 | real-valued approximation/interpolation of a discrete-valued target, such as a count (e.g., number of phone calls)                                                                |

† Provided for completeness but discouraged to avoid [ambiguities in
representation](https://github.com/alan-turing-institute/MLJ.jl/blob/dev/paper/paper.md#a-unified-approach-to-probabilistic-predictions-and-their-evaluation).

> Table of concrete subtypes of `LearnAPI.IID <: LearnAPI.KindOfProxy`.


## When the proxy for the target is a single object

In the following table of subtypes `T <: LearnAPI.KindOfProxy` not falling under the `IID`
umbrella, it is understood that `predict(model, ::T, ...)` is
not divided into individual observations, but represents a *single* probability
distribution for the sample space ``Y^n``, where ``Y`` is the space the target variable
takes its values, and `n` is the number of observations in `data`.

| type `T`                        | form of output of `predict(model, ::T, data...)`                                                                                                                     |
|:-------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LearnAPI.JointSampleable`      | object that can be sampled to obtain a *vector* whose elements have the form of target observations; the vector length matches the number of observations in `data`. |
| `LearnAPI.JointDistribution`    | explicit probability density/mass function whose sample space is vectors of target observations;  the vector length matches the number of observations in `data`     |
| `LearnAPI.JointLogDistribution` | explicit log-probability density/mass function whose sample space is vectors of target observations;  the vector length matches the number of observations in `data` |

> Table of `LearnAPI.KindOfProxy` subtypes not subtyping `LearnAPI.IID`
