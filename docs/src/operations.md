# [Predict and other operations](@id operations)

An *operation* is any method with signature `(model, fitted_params, data...)`, where `fitted_params`
is the learned parameters object, as returned by [`LearnAPI.fit`](@ref) (which will be
`nothing` if `fit` is not implemented). For example, `predict` in the following code snippet
is an operation:

```julia
fitted_params, state, report = LearnAPI.fit(some_model, 1, X, y)
ŷ = predict(some_model, fitted_params, Xnew)
```

## General requirements

- Each `model` must implement at least one of: `predict`, `transform`, `predict_joint`. 

- If `LearnAPI.is_supervised(model) == true` then `predict` or `predict_joint` must be
  implemented. 

- Do not overload `predict_mode`, `predict_mean` or `predict_median` unless
 `predict` has been implemented.
 
- Do not overload `inverse_transform` unless `transform` has been implemented. 

- Each operation explicitly implemented or overloaded must be included in the return value
  of [`LearnAPI.implemented_methods`](@ref).


| method                                | fallback                     |
|:--------------------------------------|:---------------------------- |
[`LearnAPI.predict`](@ref)           | none                         |
[`LearnAPI.predict_mode`](@ref)      | none †                       |
[`LearnAPI.predict_mean`](@ref)      | broadcast `Statistics.mean`  |
[`LearnAPI.predict_median`](@ref)    | broadcast `Statistic.median` |
[`LearnAPI.predict_joint`](@ref)     | none                         |
[`LearnAPI.transform`](@ref)         | none                         |
[`MLJInterface.inverse_transform`](@ref)| none                         |

> **† MLJ only.** MLJBase provides a fallback for `predict_mode`, which broadcasts 
> `StatBase.mode` over observations returned by `LearnAPI.predict`.

## Specifics

```@docs
LearnAPI.predict
LearnAPI.predict_mean
LearnAPI.predict_median
LearnAPI.predict_joint
```
