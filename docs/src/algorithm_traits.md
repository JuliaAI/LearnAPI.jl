# Algorithm Traits

> **Summary.** Traits allow one to promise particular behaviour for an algorithm, such as:
> *This algorithm supports per-observation weights, which must appear as the third
> *argument of > `fit`*, or *This algorithm predicts probability distributions for the
> *target*, or *This > algorithm's `transform` method predicts `Real` vectors*.

For any (non-trivial) algorithm, [`LearnAPI.functions`](@ref)`(algorithm)` must be
overloaded to list the LearnAPI methods that have been explicitly overloaded (algorithm
traits excluded). Otherwise, overloading traits is optional, except where required by the
implementation of some LearnAPI method and explicitly documented in that method's
docstring.

Traits are often called on instances but are usually *defined* on algorithm *types*, as in

```julia
LearnAPI.is_pure_julia(::Type{<:MyAlgorithmType}) = true
```

which has the shorthand

```julia
@trait MyAlgorithmType is_pure_julia=true
```

So, for convenience, every trait `t` is provided the fallback implementation

```julia
t(algorithm) = t(typeof(algorithm))
```

This means `LearnAPI.is_pure_julia(algorithm) = true` whenever `algorithm isa MyAlgorithmType` in the
above example.

Every trait has a global fallback implementation for `::Type`. 

Traits that vary from instance to instance of the same type are disallowed, except in the
case of composite algorithms (`is_wrapper(algorithm) = true`) where this is unavoidable. (One
reason for this is so one can associate with each algorithm type a unique set of trait-based
"algorithm metadata" for inclusion in searchable algorithm databases.) This requirement
occasionally requires that an existing algorithm implementation be split into separate
LearnAPI implementations (e.g., one for regression and another for classification).

**Ordinary traits** are available for overloading by any new LearnAPI implementation. **Derived
traits** are not.

## Ordinary traits

In the examples column of the table below, `Table`, `Continuous`, `Sampleable` are names owned by the
package [ScientificTypesBase.jl](https://github.com/JuliaAI/ScientificTypesBase.jl/).

| trait                                            | fallback value        | return value  | example |
|:-------------------------------------------------|:----------------------|:--------------|:--------|
| [`LearnAPI.functions`](@ref)`(algorithm)`            | `()`                  | implemented LearnAPI functions (traits excluded) | `(:fit, :predict)` |
| [`LearnAPI.predict_proxy`](@ref)`(algorithm)`        | `LearnAPI.None()`     | form of target proxy output by `predict` | `LearnAPI.Distribution()` |
| [`LearnAPI.predict_joint_proxy`](@ref)`(algorithm)`  | `LearnAPI.None()`     | form of target proxy output by `predict_joint` | `LearnAPI.Distribution()` |
| [`LearnAPI.position_of_target`](@ref)`(algorithm)`   | `0`                   | † the positional index of the **target** in `data` in `fit(..., data...; metadata)` calls | 2 |
| [`LearnAPI.position_of_weights`](@ref)`(algorithm)`  | `0`                   | † the positional index of **per-observation weights** in `data` in `fit(..., data...; metadata)` | 3 |
| [`LearnAPI.descriptors`](@ref)`(algorithm)`          | `()`                  | lists one or more suggestive algorithm descriptors from `LearnAPI.descriptors()` | (:classifier, :probabilistic) |
| [`LearnAPI.is_pure_julia`](@ref)`(algorithm)`        | `false`               | is `true` if implementation is 100% Julia code | `true` |
| [`LearnAPI.pkg_name`](@ref)`(algorithm)`             | `"unknown"`           | name of package providing core code (may be different from package providing LearnAPI.jl implementation) | `"DecisionTree"` |
| [`LearnAPI.pkg_license`](@ref)`(algorithm)`          | `"unknown"`             | name of license of package providing core code | `"MIT"` |
| [`LearnAPI.doc_url`](@ref)`(algorithm)`               | `"unknown"`             | url providing documentation of the core code  | `"https://en.wikipedia.org/wiki/Decision_tree_learning"` |
| [`LearnAPI.load_path`](@ref)`(algorithm)`            | `"unknown"`             | a string indicating where the struct for `typeof(algorithm)` is defined, beginning with name of package providing implementation | `FastTrees.LearnAPI.DecisionTreeClassifier` |
| [`LearnAPI.is_wrapper`](@ref)`(algorithm)`          | `false`                | is `true` if one or more properties (fields) of `algorithm` may be an algorithm | `true` |
| [`LearnAPI.human_name`](@ref)`(algorithm)`          | type name with spaces  | human name for the algorithm; should be a noun | "elastic net regressor" |
| [`LearnAPI.iteration_parameter`](@ref)`(algorithm)` | `nothing`                | symbolic name of an iteration parameter | :epochs |
| [`LearnAPI.fit_keywords`](@ref)`(algorithm)`        |  `()`                  | tuple of symbols for keyword arguments accepted by `fit` (corresponding  to metadata) | `(:class_weights,)` |
| [`LearnAPI.fit_scitype`](@ref)`(algorithm)`      | `Union{}` | upper bound on `scitype(data)` in `fit(algorithm, verbosity, data...)`†† | `Tuple{Table(Continuous), AbstractVector{Continuous}}` |
| [`LearnAPI.fit_observation_scitype`](@ref)`(algorithm)` | `Union{}`| upper bound on `scitype(observation)` for `observation` in `data` and `data` in `fit(algorithm, verbosity, data...)`†† | `Tuple{AbstractVector{Continuous}, Continuous}` |
| [`LearnAPI.fit_type`](@ref)`(algorithm)`            | `Union{}` | upper bound on `type(data)` in `fit(algorithm, verbosity, data...)`†† | `Tuple{AbstractMatrix{<:Real}, AbstractVector{<:Real}}` |
| [`LearnAPI.fit_observation_type`](@ref)`(algorithm)`    | `Union{}`| upper bound on `type(observation)` for `observation` in `data` and `data` in `fit(algorithm, verbosity, data...)`*    | `Tuple{AbstractVector{<:Real}, Real}` |
| [`LearnAPI.predict_input_scitype`](@ref)`(algorithm)`  | `Union{}` | upper bound on `scitype(data)` in `predict(algorithm, fitted_params, data...)`††   | `Table(Continuous)` |
| [`LearnAPI.predict_output_scitype`](@ref)`(algorithm)` | `Any`     | upper bound on `scitype(first(predict(algorithm, ...)))`                          | `AbstractVector{Continuous}` |
| [`LearnAPI.predict_input_type`](@ref)`(algorithm)`     | `Union{}` | upper bound on `typeof(data)` in `predict(algorithm, fitted_params, data...)`††    | `AbstractMatrix{<:Real}` |
| [`LearnAPI.predict_output_type`](@ref)`(algorithm)`    | `Any`     | upper bound on `typeof(first(predict(algorithm, ...)))`                           | `AbstractVector{<:Real}` |
| [`LearnAPI.predict_joint_input_scitype`](@ref)`(algorithm)`  | `Union{}` | upper bound on `scitype(data)` in `predict_joint(algorithm, fitted_params, data...)`††   |`Table(Continuous)` |
| [`LearnAPI.predict_joint_output_scitype`](@ref)`(algorithm)` | `Any`     | upper bound on `scitype(first(predict_joint(algorithm, ...)))`                          | `Sampleable{<:AbstractVector{Continuous}}` |
| [`LearnAPI.predict_joint_input_type`](@ref)`(algorithm)`     | `Union{}` | upper bound on `typeof(data)` in `predict_joint(algorithm, fitted_params, data...)`††    | `AbstractMatrix{<:Real}` |
| [`LearnAPI.predict_joint_output_type`](@ref)`(algorithm)`    | `Any`     | upper bound on `typeof(first(predict_joint(algorithm, ...)))`                           | `Distributions.Sampleable{Distributions.Multivariate,Distributions.Continuous}` |
| [`LearnAPI.transform_input_scitype`](@ref)`(algorithm)`  | `Union{}` | upper bound on `scitype(data)` in `transform(algorithm, fitted_params, data...)`††   | `Table(Continuous)` |
| [`LearnAPI.transform_output_scitype`](@ref)`(algorithm)` | `Any`     | upper bound on `scitype(first(transform(algorithm, ...)))`                          |  `Table(Continuous)` |
| [`LearnAPI.transform_input_type`](@ref)`(algorithm)`     | `Union{}` | upper bound on `typeof(data)` in `transform(algorithm, fitted_params, data...)`††    | `AbstractMatrix{<:Real}}` |
| [`LearnAPI.transform_output_type`](@ref)`(algorithm)`    | `Any`     | upper bound on `typeof(first(transform(algorithm, ...)))`                           | `AbstractMatrix{<:Real}` |
| [`LearnAPI.inverse_transform_input_scitype`](@ref)`(algorithm)`  | `Union{}` | upper bound on `scitype(data)` in `inverse_transform(algorithm, fitted_params, data...)`††  | `Table(Continuous)` |
| [`LearnAPI.inverse_transform_output_scitype`](@ref)`(algorithm)` | `Any`     | upper bound on `scitype(first(inverse_transform(algorithm, ...)))`                          | `Table(Continuous)` |
| [`LearnAPI.inverse_transform_input_type`](@ref)`(algorithm)`     | `Union{}` | upper bound on `typeof(data)` in `inverse_transform(algorithm, fitted_params, data...)`††   | `AbstractMatrix{<:Real}` |
| [`LearnAPI.inverse_transform_output_type`](@ref)`(algorithm)`    | `Any`     | upper bound on `typeof(first(inverse_transform(algorithm, ...)))`                           | `AbstractMatrix{<:Real}` |


† If the value is `0`, then the variable in boldface type is not supported and not
expected to appear in `data`. If `length(data)` is less than the trait value, then `data`
is understood to exclude the variable, but note that `fit` can have multiple signatures of
varying lengths, as in `fit(algorithm, verbosity, X, y)` and `fit(algorithm, verbosity, X, y,
w)`. A non-zero value is a promise that `fit` includes a signature of sufficient length to
include the variable.

†† Assuming no [optional data interface](@ref data_interface) is implemented. See docstring
for the general case.


## Derived Traits

The following convenience methods are provided but intended for overloading:

| trait                        | return value                              | example |
|:-----------------------------|:------------------------------------------|:--------|
| `LearnAPI.name(algorithm)`       | algorithm type name as string                 | "PCA"   |
| `LearnAPI.is_algorithm(algorithm)`   | `true` if `functions(algorithm)` is not empty | `true`  |

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
