# [Kinds of Target Proxy](@id proxy_types)

The available kinds of [target proxy](@ref proxy) (used for `predict` dispatch) are
classified by subtypes of `LearnAPI.KindOfProxy`. These types are intended for dispatch
only and have no fields.

```@docs
LearnAPI.KindOfProxy
```

## Simple target proxies

```@docs
LearnAPI.IID
```

| type                                  | form of an observation                                                                                                                                                            |
|:-------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LearnAPI.LiteralTarget`              | same as target observations; may have the interpretation of a 50% quantile, 50% expectile or mode                                                                                 |
| `LearnAPI.Sampleable`                 | object that can be sampled to obtain object of the same form as target observation                                                                                                |
| `LearnAPI.Distribution`               | explicit probability density/mass function whose sample space is all possible target observations                                                                                 |
| `LearnAPI.LogDistribution`            | explicit log-probability density/mass function whose sample space is possible target observations                                                                                 |
| `LearnAPI.Probability`¹               | numerical probability or probability vector                                                                                                                                       |
| `LearnAPI.LogProbability`¹            | log-probability or log-probability vector                                                                                                                                         |
| `LearnAPI.Parametric`¹                | a list of parameters (e.g., mean and variance) describing some distribution                                                                                                       |
| `LearnAPI.LabelAmbiguous`             | collections of labels (in case of multi-class target) but without a known correspondence to the original target labels (and of possibly different number) as in, e.g., clustering |
| `LearnAPI.LabelAmbiguousSampleable`   | sampleable version of `LabelAmbiguous`; see `Sampleable` above                                                                                                                    |
| `LearnAPI.LabelAmbiguousDistribution` | pdf/pmf version of `LabelAmbiguous`; see `Distribution`  above                                                                                                                    |
| `LearnAPI.LabelAmbiguousFuzzy`        | same as `LabelAmbiguous` but with multiple values of indeterminant number                                                                                                         |
| `LearnAPI.Quantile`²                  | same as target but with quantile interpretation                                                                                                                                   |
| `LearnAPI.Expectile`²                 | same as target but with expectile interpretation                                                                                                                                  |
| `LearnAPI.ConfidenceInterval`²        | confidence interval                                                                                                                                                               |
| `LearnAPI.Fuzzy`                      | finite but possibly varying number of target observations                                                                                                                         |
| `LearnAPI.ProbabilisticFuzzy`         | as for `Fuzzy` but labeled with probabilities (not necessarily summing to one)                                                                                                    |
| `LearnAPI.SurvivalFunction`           | survival function                                                                                                                                                                 |
| `LearnAPI.SurvivalDistribution`       | probability distribution for survival time                                                                                                                                        |
| `LearnAPI.SurvivalHazardFunction`     | hazard function for survival time                                                                                                                                                 |
| `LearnAPI.OutlierScore`               | numerical score reflecting degree of outlierness (not necessarily normalized)                                                                                                     |
| `LearnAPI.Continuous`                 | real-valued approximation/interpolation of a discrete-valued target, such as a count (e.g., number of phone calls)                                                                |

¹Provided for completeness but discouraged to avoid [ambiguities in
representation](https://github.com/alan-turing-institute/MLJ.jl/blob/dev/paper/paper.md#a-unified-approach-to-probabilistic-predictions-and-their-evaluation).

²The level will be controlled by a hyper-parameter; models providing only quantiles or
expectiles at 50% will provide `LiteralTarget` instead.

> Table of concrete subtypes of `LearnAPI.IID <: LearnAPI.KindOfProxy`.


## Proxies for density estimation lgorithms

```@docs
LearnAPI.Single
```

| type `T`                         | form of output of `predict(model, ::T)`                                |
|:--------------------------------:|:-----------------------------------------------------------------------|
| `LearnAPI.SingleSampleable`      | object that can be sampled to obtain a single target observation       |
| `LearnAPI.SingleDistribution`    | explicit probability density/mass function for sampling the target     |
| `LearnAPI.SingleLogDistribution` | explicit log-probability density/mass function for sampling the target |

> Table of `LearnAPI.KindOfProxy` subtypes subtyping `LearnAPI.Single`


## Joint probability distributions

```@docs
LearnAPI.Joint
```

| type `T`                        | form of output of `predict(model, ::T, data)`                                                                                                                     |
|:-------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LearnAPI.JointSampleable`      | object that can be sampled to obtain a *vector* whose elements have the form of target observations; the vector length matches the number of observations in `data`. |
| `LearnAPI.JointDistribution`    | explicit probability density/mass function whose sample space is vectors of target observations;  the vector length matches the number of observations in `data`     |
| `LearnAPI.JointLogDistribution` | explicit log-probability density/mass function whose sample space is vectors of target observations;  the vector length matches the number of observations in `data` |

> Table of `LearnAPI.KindOfProxy` subtypes subtyping `LearnAPI.Joint`
