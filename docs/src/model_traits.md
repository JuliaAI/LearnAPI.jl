# Model Traits

| trait                                            | fallback value        | return value  | example |
|:-------------------------------------------------|:----------------------|:--------------|:--------|
| [`LearnAPI.ismodel`](@ref)`(model)`              | `false`               | is `true` for any model, as defined in [`Models`](@ref) | `true` |
| [`LearnAPI.implemented_methods`](@ref)`(model)`  | `()`                  | lists of all overloaded/implemented methods (traits excluded) | `(:fit, :predict)` |
| [`LearnAPI.target_proxy_kind`](@ref)`(model)`    | `()`                  | details form of target proxy output | `(predict= LearnAPI.Distribution,)` |
| [`LearnAPI.position_of_target`](@ref)`(model)`   | `0`                   | † the positional index of the **target** in `data` in `fit(..., data...; metadata)` calls | 2 |
| [`LearnAPI.position_of_weights`](@ref)`(model)`  | `0`                   | † the positional index of **observation weights** in `data` in `fit(..., data...; metadata)` | 3 |
| [`LearnAPI.descriptors`](@ref)`(model)`          | `()`                  | lists one or more suggestive model descriptors from `LearnAPI.descriptors()` | (:classifier, :probabilistic) |
| [`LearnAPI.is_pure_julia`](@ref)`(model)`        | `false`               | is `true` if implementation is 100% Julia code | `true` |
| [`LearnAPI.pkg_name`](@ref)`(model)`             | "unknown"             | name of package providing core algorithm (may be different from package providing LearnAPI.jl implementation) | "DecisionTree" |
| [`LearnAPI.doc_url`](@ref)`(model)               | "unknown"             | url providing documentation of the core algorithm  | "https://en.wikipedia.org/wiki/Decision_tree_learning" |
| [`LearnAPI.pkg_license`](@ref)`(model)`          | "unknown"             | name of license of package providing core algorithm | "MIT" |
| [`LearnAPI.load_path`](@ref)`(model)`            | "unknown"             | a string indicating where the struct `typeof(model)` is defined, beginning with name of package providing implementation | `FastTrees.LearnAPI.DecisionTreeClassifier` |
| [`LearnAPI.is_wrapper`](@ref)`(model)`          | `false`                | is `true` if one or more hyperparameters are themselves models | `true` |
| [`LearnAPI.fit_keywords`](@ref)`(model)`        |  `()`                  | tuple of symbols for keyword arguments accepted by `fit` (metadata) | `(:class_weights,)` |
| [`LearnAPI.human_name`](@ref)`(model)`          | type name with spaces, acronymns preserved  | human name for the model; should be a noun | "elastic net regressor" |
| [`LearnAPI.iteration_parameter`](@ref)`(model)` | nothing                | symbolic name of an iteration parameter | :epochs |


† If the value is `0`, then the variable in boldface type is not supported and not
expected to appear in `data`. If `length(data)` is less than the trait value, then `data`
is understood to exclude the variable, but note that `fit` can have multiple signatures of
varying lengths, as in `fit(model, verbosity, X, y)` and `fit(model, verbosity, X, y,
w)`. A non-zero value is a promise that `fit` includes a signature of sufficient length to
include the variable.
