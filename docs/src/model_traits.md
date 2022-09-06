# Model Traits

| trait                                            | fallback value        | return value  | example |
|:-------------------------------------------------|:----------------------|:--------------|:--------|
| [`LearnAPI.ismodel`](@ref)`(model)`              | `false`               | is `true` for any model, as defined in [`Models`](@ref) | `true` |
| [`LearnAPI.implemented_methods`](@ref)`(model)`  | `()`                  | lists of all overloaded/implemented methods (traits excluded) | `(:fit, :predict)` |
| [`LearnAPI.target_proxy_kind`](@ref)`(model)`    | `()`                  | details form of target proxy output | `(predict= LearnAPI.Distribution,)` |
| [`LearnAPI.position_of_target`](@ref)`(model)`   | `0`                   | † the positional index of the **target** in `data` in `fit(..., data...; metadata)` calls | 2 |
| [`LearnAPI.position_of_weights`](@ref)`(model)`  | `0`                   | † the positional index of **observation weights** in `data` in `fit(..., data...; metadata)` | 3 |
| [`LearnAPI.keywords`](@ref)`(model)`             | `()`                  | lists one or more suggestive model descriptors from `LearnAPI.keywords()` | (:regressor, ) |

† If the value is `0`, then the variable in boldface type is not supported and never
appears in `data`. If `length(data)` exceeds the trait value, then `data` is understood to
exclude the variable, but note that `fit` can have multiple signatures of varying lengths,
as in `fit(model, verbosity, X, y)` and `fit(model, verbosity, X, y, w)`. A non-zero value
is a promise that `fit` includes a signature where the variable is included.
