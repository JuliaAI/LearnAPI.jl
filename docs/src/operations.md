# [Predict and other operations](@id operations)

> **Summary** Methods like `predict` and `transform`, that generally depend on learned
> parameters, are called **operations**. All implemented operations must be included in
> the output of the `implemented_methods` model trait. When an operation returns a [target
> proxy](@ref scope), it must make a `target_proxy_kind` declaration.

An *operation* is any method with signature `some_operation(model, fitted_params,
data...)`. Here `fitted_params` is the learned parameters object, as returned by
[`LearnAPI.fit`](@ref)`(model, ...)`, which will be `nothing` if `fit` is not implemented
(true for models that do not generalize to new data). For example, `LearnAPI.predict` in
the following code snippet is an operation:

```julia
fitted_params, state, fit_report = LearnAPI.fit(some_model, 1, X, y)
ŷ, predict_report = LearnAPI.predict(some_model, fitted_params, Xnew)
```

| method                             | compulsory? | fallback | requires    |
|:-----------------------------------|:-----------:|:--------:|:-----------:|
[`LearnAPI.predict`](@ref)           | no          | none     |             |
[`LearnAPI.predict_mode`](@ref)      | no          | none     | `predict`   |
[`LearnAPI.predict_mean`](@ref)      | no          | none     | `predict`   |
[`LearnAPI.predict_median`](@ref)    | no          | none     | `predict`   |
[`LearnAPI.predict_joint`](@ref)     | no          | none     |             |
[`LearnAPI.transform`](@ref)         | no          | none     |             |
[`LearnAPI.inverse_transform`](@ref) | no          | none     | `transform` |

> **† MLJ only.** MLJBase provides fallbacks for `predict_mode`, `predict_mean` and
> `predict_median` by broadcasting methods from `Statistics` and `StatsBase` over the
> results of `predict`.

## General requirements

- Only implement `predict_joint` for outputing a *single* multivariate probability
  distribution for multiple target predictions, as described further at
  [`LearnAPI.predict_joint`](@ref).

- Each operation explicitly implemented or overloaded must be included in the return value
  of [`LearnAPI.implemented_methods`](@ref).

## Predict or transform?

- If the model has a target, as defined under [Scope and undefined notions](@ref scope), then
  only `predict` or `predict_joint` can be used to generate a corresponding target proxy.

- If an operation is to have an inverse operation, then it cannot be `predict` - use
  `transform` and `inverse_transform`.

- If only a single operation is implemented, and there is no target variable, use `transform`. 

Here an "inverse" of `transform` is very broadly understood as any operation that can be
applied to the output of `transform` to obtain an object of the same form as the input of
`transform`; for example this includes one-sided inverses, and approximate one-sided
inverses. 


## Target proxies

In the case that a model has the concept of a **target** variable, as described under
[Scope and undefined notions](@ref scope), the output of `predict` or `predict_joint` may
have the form of a proxy for the target, such as a vector of truth-probabilities for
binary targets.

We assume the reader is already familiar with the notion of a target variable in
supervised learning, but target variables are not limited to supervised models. For
example, we may regard the "outlier"/"inlier" assignments in unsupervised anomaly
detection as a target. A target proxy in this example would be probabilities for
outlierness, as these can be paired with "outlier"/"inlier" labels assigned by humans,
using, say, area under the ROC curve, to quantify performance.

Similarly, the integer labels assigned to some observations by a clustering algorithm can
be regarded as a target variable. The labels obtained can be paired with human labels
using, say, the Rand index. 

The kind of proxy one has is informally classified by a subtype of
`LearnAPI.TargetProxy`. These types are intended for dispatch outside of LearnAPI.jl and
have no fields.

|          type                   | form of observations | possible requirement in some external API |
|:-------------------------------:|:---------------------|:------------------------------------------|
| `LearnAPI.TrueTarget`              | same as target observations | Observations have same type as target observations. |
| `LearnAPI.Sampleable`           | objects that can be sampled to obtain objects of the same form as target observations | Each observation implements `Base.rand`. |
| `LearnAPI.Distribution`         | explicit probability density/mass functions with sample space all possible target observations | Observations implement `Distributions.pdf` and `Base.rand` |
| `LearnAPI.LogDistribution`      | explicit log probability density/mass functions with sample space all possible target observations | Observations implement `Distributions.logpdf` and `Base.rand` |
|  † `LearnAPI.Probability`       | raw numerical probability or probability vector | |
|  † `LearnAPI.LogProbability`    | log probability or log probability vector | |
|  † `LearnAPI.Parametric`        | a list of parameters (e.g., mean and variance) describing some distribution |
| `LearnAPI.LabelAmbiguous`            | same form as the (multi-class) target, but with new, unmatched labels of possibly unequal number (as in, e.g., clustering)| 
| `LearnAPI.LabelAmbiguousSampleable`  | sampleable version of `LabelAmbiguous`; see `Sampleable` above  |
| `LearnAPI.LabelAmbiguousDistribution`| pdf/pmf version of `LabelAmbiguous`; see `Distribution`  above  |
| `LearnAPI.ConfidenceInterval`   | confidence intervals |  Each observation `isa Tuple{Real,Real}`.
| `LearnAPI.SurvivalFunction`     | survival functions | Observations are single-argument functions mapping `Real` to `Real`.
| `LearnAPI.SurvivalDistribution` | probability distribution for survival time | Observations have type `Distributions.ContinuousUnivariateDistribution`.

> **† MLJ only.** To avoid [ambiguities in
> representation](https://github.com/alan-turing-institute/MLJ.jl/blob/dev/paper/paper.md#a-unified-approach-to-probabilistic-predictions-and-their-evaluation),
> these options are disallowed, in favour of the preceding alternatives.

!!! warning

	The last column of the table is not part of LearnAPI.jl.

An operation with target proxy as output must declare the `TargetProxy` subtype using the
[`LearnAPI.target_proxy_kind`](@ref), as in

```julia
LearnAPI.target_proxy_kind(::Type{<:SomeModel}) = (predict=LearnAPI.Distribution,)
```

### Special case of predict_joint

If `predict_joint` is implemented, then a `target_proxy_kind` declaration is required, but
the interpretation is slightly different. This is because the output of `predict_joint` is
not a number of observations but a single object. See more at [`LearnAPI.predict_joint`](@ref) below.


## Operation-specific details

```@docs
LearnAPI.predict
LearnAPI.predict_mean
LearnAPI.predict_median
LearnAPI.predict_joint
LearnAPI.transform
LearnAPI.inverse_transform
```