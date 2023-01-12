# Model Traits

Traits are often called on instances but are frequently *defined* on model *types*, as in 

```julia
LearnAPI.is_pure_julia(::Type{<:MyModelType}) = true
```

which has the shorthand

```julia
@trait MyModelType is_pure_julia=true
```

So, for convenience, every trait `t` is provided the fallback implementation

```julia
t(model) = t(typeof(model))
```

This means `LearnAPI.is_pure_julia(model) = true` whenever `model isa MyModelType` in the
above example. 

Traits that vary from instance to instance of the same type are discouraged, except in
the case of composite models (`is_wrapper(model) = true`) where this is unavoidable. One
reason for this so one can associate with each model type a unique set of trait-based
"model metadata" for inclusion in searchable model databases. This requirement
occasionally requires that an existing model implementation be split into several separate
LearnAPI implementations (e.g., one for regression and another for classification).

Ordinary traits are available for overloading by an new model implementation. Derived
traits are not.

## Ordinary traits

In the examples column of the table below, `Table` and `Continuous` are names owned by the
package [ScientificTypesBase.jl](https://github.com/JuliaAI/ScientificTypesBase.jl/).

| trait                                            | fallback value        | return value  | example |
|:-------------------------------------------------|:----------------------|:--------------|:--------|
| [`LearnAPI.functions`](@ref)`(model)`            | `()`                  | implemented LearnAPI functions (traits excluded) | `(:fit, :predict)` |
| [`LearnAPI.predict_proxy`](@ref)`(model)`        | `NamedTuple()`        | form of target proxy output by `predict` | `LearnAPI.Distribution()` |
| [`LearnAPI.predict_joint_proxy`](@ref)`(model)`  | `NamedTuple()`        | form of target proxy output by `predict_joint` | `LearnAPI.Distribution()` |
| [`LearnAPI.position_of_target`](@ref)`(model)`   | `0`                   | † the positional index of the **target** in `data` in `fit(..., data...; metadata)` calls | 2 |
| [`LearnAPI.position_of_weights`](@ref)`(model)`  | `0`                   | † the positional index of **per-observation weights** in `data` in `fit(..., data...; metadata)` | 3 |
| [`LearnAPI.descriptors`](@ref)`(model)`          | `()`                  | lists one or more suggestive model descriptors from `LearnAPI.descriptors()` | (:classifier, :probabilistic) |
| [`LearnAPI.is_pure_julia`](@ref)`(model)`        | `false`               | is `true` if implementation is 100% Julia code | `true` |
| [`LearnAPI.pkg_name`](@ref)`(model)`             | `"unknown"`           | name of package providing core algorithm (may be different from package providing LearnAPI.jl implementation) | `"DecisionTree"` |
| [`LearnAPI.pkg_license`](@ref)`(model)`          | `"unknown"`             | name of license of package providing core algorithm | `"MIT"` |
| [`LearnAPI.doc_url`](@ref)`(model)`               | `"unknown"`             | url providing documentation of the core algorithm  | `"https://en.wikipedia.org/wiki/Decision_tree_learning"` |
| [`LearnAPI.load_path`](@ref)`(model)`            | `"unknown"`             | a string indicating where the struct `typeof(model)` is defined, beginning with name of package providing implementation | `FastTrees.LearnAPI.DecisionTreeClassifier` |
| [`LearnAPI.is_wrapper`](@ref)`(model)`          | `false`                | is `true` if one or more properties (fields) are themselves models | `true` |
| [`LearnAPI.human_name`](@ref)`(model)`          | type name with spaces  | human name for the model; should be a noun | "elastic net regressor" |
| [`LearnAPI.iteration_parameter`](@ref)`(model)` | nothing                | symbolic name of an iteration parameter | :epochs |
| [`LearnAPI.fit_keywords`](@ref)`(model)`        |  `()`                  | tuple of symbols for keyword arguments accepted by `fit` (metadata) | `(:class_weights,)` |
| [`LearnAPI.fit_scitype`](@ref)`(model)`      | `Union{}` | upper bound on `scitype(data)` in `fit(model, verbosity, data...)`†† | `Tuple{Table(Continuous), AbstractVector{Continuous}}` |
| [`LearnAPI.fit_type`](@ref)`(model)`            | `Union{}` | upper bound on `type(data)` in `fit(model, verbosity, data...)`†† | `Tuple{AbstractMatrix{<:Real}, AbstractVector{<:Real}}` |
| [`LearnAPI.fit_observation_scitype`](@ref)`(model)` | `Union{}`| upper bound on `scitype(data)` in `fit(model, verbosity, data...)`†† | `Tuple{AbstractVector{Continuous}, Continuous}` |
| [`LearnAPI.fit_observation_type`](@ref)`(model)`    | `Union{}`| upper bound on `type(data)` in `fit(model, verbosity, data...)`*    | `Tuple{AbstractVector{<:Real}, Real}` |
| [`LearnAPI.predict_input_scitype`](@ref)`(model)`  | `Union{}` | upper bound on `scitype(data)` in `predict(model, fitted_params, data...)`††   | `Tuple{AbstractVector{Continuous}}` |
| [`LearnAPI.predict_output_scitype`](@ref)`(model)` | `Any`     | upper bound on `scitype(first(predict(model, ...)))`                          | `AbstractVector{Continuous}` |
| [`LearnAPI.predict_input_type`](@ref)`(model)`     | `Union{}` | upper bound on `typeof(data)` in `predict(model, fitted_params, data...)`††    | `Tuple{AbstractVector{<:Real}}` |
| [`LearnAPI.predict_output_type`](@ref)`(model)`    | `Any`     | upper bound on `typeof(first(predict(model, ...)))`                           | `AbstractVector{<:Real}` |
| [`LearnAPI.predict_joint_input_scitype`](@ref)`(model)`  | `Union{}` | upper bound on `scitype(data)` in `predict_joint(model, fitted_params, data...)`††   | `Tuple{AbstractVector{Continuous}}` |
| [`LearnAPI.predict_joint_output_scitype`](@ref)`(model)` | `Any`     | upper bound on `scitype(first(predict_joint(model, ...)))`                          | `AbstractVector{Continuous}` |
| [`LearnAPI.predict_joint_input_type`](@ref)`(model)`     | `Union{}` | upper bound on `typeof(data)` in `predict_joint(model, fitted_params, data...)`††    | `Tuple{AbstractVector{<:Real}}` |
| [`LearnAPI.predict_joint_output_type`](@ref)`(model)`    | `Any`     | upper bound on `typeof(first(predict_joint(model, ...)))`                           | `AbstractVector{<:Real}` |
| [`LearnAPI.transform_input_scitype`](@ref)`(model)`  | `Union{}` | upper bound on `scitype(data)` in `transform(model, fitted_params, data...)`††   | `Tuple{AbstractVector{Continuous}}` |
| [`LearnAPI.transform_output_scitype`](@ref)`(model)` | `Any`     | upper bound on `scitype(first(transform(model, ...)))`                          | `AbstractVector{Continuous}` |
| [`LearnAPI.transform_input_type`](@ref)`(model)`     | `Union{}` | upper bound on `typeof(data)` in `transform(model, fitted_params, data...)`††    | `Tuple{AbstractVector{<:Real}}` |
| [`LearnAPI.transform_output_type`](@ref)`(model)`    | `Any`     | upper bound on `typeof(first(transform(model, ...)))`                           | `AbstractVector{<:Real}` |
| [`LearnAPI.inverse_transform_input_scitype`](@ref)`(model)`  | `Union{}` | upper bound on `scitype(data)` in `inverse_transform(model, fitted_params, data...)`††   | `Tuple{AbstractVector{Continuous}}` |
| [`LearnAPI.inverse_transform_output_scitype`](@ref)`(model)` | `Any`     | upper bound on `scitype(first(inverse_transform(model, ...)))`                          | `AbstractVector{Continuous}` |
| [`LearnAPI.inverse_transform_input_type`](@ref)`(model)`     | `Union{}` | upper bound on `typeof(data)` in `inverse_transform(model, fitted_params, data...)`††    | `Tuple{AbstractVector{<:Real}}` |
| [`LearnAPI.inverse_transform_output_type`](@ref)`(model)`    | `Any`     | upper bound on `typeof(first(inverse_transform(model, ...)))`                           | `AbstractVector{<:Real}` |


† If the value is `0`, then the variable in boldface type is not supported and not
expected to appear in `data`. If `length(data)` is less than the trait value, then `data`
is understood to exclude the variable, but note that `fit` can have multiple signatures of
varying lengths, as in `fit(model, verbosity, X, y)` and `fit(model, verbosity, X, y,
w)`. A non-zero value is a promise that `fit` includes a signature of sufficient length to
include the variable.

†† Assuming no [optional data interface](@ref data_interface) is implemented. See docstring
for the general case.


## Derived Traits

The following convenience methods are provided but intended for overloading:

| trait                        | return value                              | example |
|:-----------------------------|:------------------------------------------|:--------|
| `LearnAPI.name(model)`       | model type name as string                 | "PCA"   |
| `LearnAPI.is_model(model)`   | `true` if `functions(model)` is not empty | `true`  |

## Reference

```@docs
LearnAPI.functions
LearnAPI.predict_proxy
LearnAPI.predict_joint_proxy
LearnAPI.position_of_target
LearnAPI.position_of_weights
LearnAPI.descriptors
LearnAPI.is_pure_julia
LearnAPI.pkg_name
LearnAPI.pkg_license
LearnAPI.doc_url
LearnAPI.load_path
LearnAPI.is_wrapper
LearnAPI.fit_keywords
LearnAPI.human_name
LearnAPI.iteration_parameter
LearnAPI.fit_scitype
LearnAPI.fit_type
LearnAPI.fit_observation_scitype
LearnAPI.fit_observation_type
LearnAPI.predict_input_scitype
LearnAPI.predict_output_scitype
LearnAPI.predict_input_type
LearnAPI.predict_output_type
LearnAPI.predict_joint_input_scitype
LearnAPI.predict_joint_output_scitype
LearnAPI.predict_joint_input_type
LearnAPI.predict_joint_output_type
LearnAPI.transform_input_scitype
LearnAPI.transform_output_scitype
LearnAPI.transform_input_type
LearnAPI.transform_output_type
LearnAPI.inverse_transform_input_scitype
LearnAPI.inverse_transform_output_scitype
LearnAPI.inverse_transform_input_type
LearnAPI.inverse_transform_output_type
```
