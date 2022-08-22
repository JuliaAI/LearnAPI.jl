# [Predict and other operations](@id operations)

An *operation* is any method with signature `some_operation(model, fitted_params,
data...)`. Here `fitted_params` is the learned parameters object, as returned by
[`LearnAPI.fit`](@ref), which will be `nothing` if `fit` is not implemented (true for models
that do not generalize to new data). For example, `predict` in the following code snippet is
an operation:

```julia
fitted_params, state, fit_report = LearnAPI.fit(some_model, 1, X, y)
ŷ, predict_report = predict(some_model, fitted_params, Xnew)
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

-  Each `model` must implement at least one of: `predict`, `transform`,
   `predict_joint`. 
   
- Only implement `predict_joint` for outputing a *single* multivariate probability
  distribution with a dimension for each input observation; see
  [`LearnAPI.predict_joint`](@ref) for details.

- Do not overload `predict_mode`, `predict_mean` or `predict_median` unless `predict` has
  been implemented.
 
- Do not overload `inverse_transform` unless `transform` has been implemented. 

- Each operation explicitly implemented or overloaded must be included in the return value
  of [`LearnAPI.implemented_methods`](@ref).

## Predict or transform? 

- If the model has a target, as defined under [Scope and undefined notions](@ref), then
  only `predict` or `predict_joint` can be used to generate corresponding target-like
  data.

- If an operation is to have an inverse operation, then it cannot be `predict` - use
  `transform` and `inverse_transform`.

Here an "inverse" of `transform` is very broadly understood as any operation that can be
applied to the output of `transform` to obtain an object of the same form as the input of
`transform`; for example this includes one-sided inverses, and approximate one-sided
inverses. (In some API's, such an operation is called `reconstruct`.)

In all other cases, the Learn API makes only informal stipulations on which operation to
use:

- Clustering algorithms should use `predict` *when returning cluster labels.* (For
  clusterering algorithms that perform dimension reduction, `transform` can be used.)

- Outlier detection models should return raw scores using `transform` and use `predict` for
  returning either normalized scores or  "outlier"/"inlier" classifications.


## Paradigms for target-like output

Target-like data, as defined under [Scope and undefined notions](@ref), is classified by a
**paradigm**, which is one of the abstract types appearing in the table below.

| paradigm type         | form of observations | possible requirement in some external API |
|:---------------------:|:--------------------|:------------------------------------------|
| `LearnAPI.Deterministic`    | the same form as target observations | Observations have same type as target observations. |
| `LearnAPI.Distribution`      | explicit probability/mass density functions with sample space all possible target observations | Observations implements `Distributions.pdf`. |
| `LearnAPI.Sampleable`     | objects that can be sampled to obtain objects of the same form as target observations) | Each observation implements `Base.rand`. |
| `LearnAPI.Interval`         | ordered pairs of real numbers |  Each observation `isa Tuple{Real,Real}`. 
| `LearnAPI.SurvivalFunction` | survival functions | Observations are single-argument functions mapping `Real` to `Real`.


!!! warning

	The last column of the table is not part of the Learn API.



## Operation specifics

```@docs
LearnAPI.predict
LearnAPI.predict_mean
LearnAPI.predict_median
LearnAPI.predict_joint
LearnAPI.transform
```
