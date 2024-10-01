# [Algorithm Traits](@id traits)

Traits generally promise specific algorithm behavior, such as: *This algorithm can make
point or probabilistic predictions*, *This algorithm sees a target variable in training*,
or *This algorithm's `transform` method predicts `Real` vectors*. They also record more
mundane information, such as a package license.

Algorithm traits are functions whose first (and usually only) argument is an algorithm.

### Special two-argument traits

The two-argument version of [`LearnAPI.predict_output_scitype`](@ref) and
[`LearnAPI.predict_output_scitype`](@ref) are the only overloadable traits with more than
one argument.

## [Trait summary](@id trait_summary)

### [Overloadable traits](@id traits_list)

In the examples column of the table below, `Table`, `Continuous`, `Sampleable` are names owned by the
package [ScientificTypesBase.jl](https://github.com/JuliaAI/ScientificTypesBase.jl/).

| trait                                                                 | return value                                                                                                                     | fallback value                                        | example                                                  |
|:----------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------|:---------------------------------------------------------|
| [`LearnAPI.constructor`](@ref)`(algorithm)`                           | constructor for generating new or modified versions of `algorithm`                                                               | (no fallback)                                         | `RidgeRegressor`                                         |
| [`LearnAPI.functions`](@ref)`(algorithm)`                             | functions you can apply to `algorithm` or associated model (traits excluded)                                                     | `()`                                                  | `(:fit, :predict, :minimize, :(LearnAPI.algorithm), :obs)`      |
| [`LearnAPI.kinds_of_proxy`](@ref)`(algorithm)`                        | instances `kind` of `KindOfProxy` for which an implementation of `LearnAPI.predict(algorithm, kind, ...)` is guaranteed.         | `()`                                                  | `(Distribution(), Interval())`                           |
| [`LearnAPI.tags`](@ref)`(algorithm)`                           | lists one or more suggestive algorithm tags from `LearnAPI.tags()`                                                 | `()`                                                  | (:regression, :probabilistic)                            |
| [`LearnAPI.is_pure_julia`](@ref)`(algorithm)`                         | `true` if implementation is 100% Julia code                                                                                      | `false`                                               | `true`                                                   |
| [`LearnAPI.pkg_name`](@ref)`(algorithm)`                              | name of package providing core code (may be different from package providing LearnAPI.jl implementation)                         | `"unknown"`                                           | `"DecisionTree"`                                         |
| [`LearnAPI.pkg_license`](@ref)`(algorithm)`                           | name of license of package providing core code                                                                                   | `"unknown"`                                           | `"MIT"`                                                  |
| [`LearnAPI.doc_url`](@ref)`(algorithm)`                               | url providing documentation of the core code                                                                                     | `"unknown"`                                           | `"https://en.wikipedia.org/wiki/Decision_tree_learning"` |
| [`LearnAPI.load_path`](@ref)`(algorithm)`                             | a string indicating where the struct for `typeof(algorithm)` is defined, beginning with name of package providing implementation | `"unknown"`                                           | `FastTrees.LearnAPI.DecisionTreeClassifier`              |
| [`LearnAPI.is_composite`](@ref)`(algorithm)`                          | `true` if one or more properties (fields) of `algorithm` may be an algorithm                                                     | `false`                                               | `true`                                                   |
| [`LearnAPI.human_name`](@ref)`(algorithm)`                            | human name for the algorithm; should be a noun                                                                                   | type name with spaces                                 | "elastic net regressor"                                  |
| [`LearnAPI.data_interface`](@ref)`(algorithm)`                        | Interface implemented by objects returned by [`obs`](@ref)                                                                       | `Base.HasLength()` (supports `MLUtils.getobs/numobs`) | `Base.SizeUnknown()` (supports `iterate`)                |
| [`LearnAPI.iteration_parameter`](@ref)`(algorithm)`                   | symbolic name of an iteration parameter                                                                                          | `nothing`                                             | :epochs                                                  |
| [`LearnAPI.fit_scitype`](@ref)`(algorithm)`                           | upper bound on `scitype(data)` ensuring `fit(algorithm, data)` works                                                             | `Union{}`                                             | `Tuple{Table(Continuous), AbstractVector{Continuous}}`   |
| [`LearnAPI.fit_observation_scitype`](@ref)`(algorithm)`               | upper bound on `scitype(observation)` for `observation` in `data` ensuring `fit(algorithm, data)` works                          | `Union{}`                                             | `Tuple{AbstractVector{Continuous}, Continuous}`          |
| [`LearnAPI.fit_type`](@ref)`(algorithm)`                              | upper bound on `typeof(data)` ensuring `fit(algorithm, data)` works                                                              | `Union{}`                                             | `Tuple{AbstractMatrix{<:Real}, AbstractVector{<:Real}}`  |
| [`LearnAPI.fit_observation_type`](@ref)`(algorithm)`                  | upper bound on `typeof(observation)` for `observation` in `data` ensuring `fit(algorithm, data)` works                           | `Union{}`                                             | `Tuple{AbstractVector{<:Real}, Real}`                    |
| [`LearnAPI.target_observation_scitype`](@ref)`(algorithm)`            | upper bound on the scitype of each observation of the targget                                                                    | `Any`                                                 | `Continuous`                                             |
| [`LearnAPI.predict_input_scitype`](@ref)`(algorithm)`                 | upper bound on `scitype(data)` ensuring `predict(model, kind, data)` works                                                       | `Union{}`                                             | `Table(Continuous)`                                      |
| [`LearnAPI.predict_input_observation_scitype`](@ref)`(algorithm)`     | upper bound on `scitype(observation)` for `observation` in `data` ensuring `predict(model, kind, data)` works                    | `Union{}`                                             | `Vector{Continuous}`                                     |
| [`LearnAPI.predict_input_type`](@ref)`(algorithm)`                    | upper bound on `typeof(data)` ensuring `predict(model, kind, data)` works                                                        | `Union{}`                                             | `AbstractMatrix{<:Real}`                                 |
| [`LearnAPI.predict_input_observation_type`](@ref)`(algorithm)`        | upper bound on `typeof(observation)` for `observation` in `data` ensuring `predict(model, kind, data)` works                     | `Union{}`                                             | `Vector{<:Real}`                                         |
| [`LearnAPI.predict_output_scitype`](@ref)`(algorithm, kind_of_proxy)` | upper bound on `scitype(predict(model, ...))`                                                                                    | `Any`                                                 | `AbstractVector{Continuous}`                             |
| [`LearnAPI.predict_output_type`](@ref)`(algorithm, kind_of_proxy)`    | upper bound on `typeof(predict(model, ...))`                                                                                     | `Any`                                                 | `AbstractVector{<:Real}`                                 |
| [`LearnAPI.transform_input_scitype`](@ref)`(algorithm)`               | upper bound on `scitype(data)` ensuring  `transform(model, data)` works                                                          | `Union{}`                                             | `Table(Continuous)`                                      |
| [`LearnAPI.transform_input_observation_scitype`](@ref)`(algorithm)`   | upper bound on `scitype(observation)` for `observation` in `data` ensuring `transform(model, data)` works                        | `Union{}`                                             | `Vector{Continuous}`                                     |
| [`LearnAPI.transform_input_type`](@ref)`(algorithm)`                  | upper bound on `typeof(data)`ensuring `transform(model, data)` works                                                             | `Union{}`                                             | `AbstractMatrix{<:Real}}`                                |
| [`LearnAPI.transform_input_observation_type`](@ref)`(algorithm)`      | upper bound on `typeof(observation)` for `observation` in `data` ensuring `transform(model, data)` works                         | `Union{}`                                             | `Vector{Continuous}`                                     |
| [`LearnAPI.transform_output_scitype`](@ref)`(algorithm)`              | upper bound on `scitype(transform(model, ...))`                                                                                  | `Any`                                                 | `Table(Continuous)`                                      |
| [`LearnAPI.transform_output_type`](@ref)`(algorithm)`                 | upper bound on `typeof(transform(model, ...))`                                                                                   | `Any`                                                 | `AbstractMatrix{<:Real}`                                 |
| [`LearnAPI.predict_or_transform_mutates`](@ref)`(algorithm)`          | `true` if `predict` or `transform` mutates first argument                                                                        | `false`                                               | `true`                                                   |

### Derived Traits

The following convenience methods are provided but not overloadable by new implementations.

| trait                                                | return value                                                                                                  | example |
|:-----------------------------------------------------|:--------------------------------------------------------------------------------------------------------------|:--------|
| `LearnAPI.name(algorithm)`                           | algorithm type name as string                                                                                 | "PCA"   |
| `LearnAPI.is_algorithm(algorithm)`                   | `true` if `LearnAPI.functions(algorithm)` is not empty                                                                 | `true`  |
| [`LearnAPI.predict_output_scitype(algorithm)`](@ref) | dictionary of upper bounds on the scitype of predictions, keyed on subtypes of [`LearnAPI.KindOfProxy`](@ref) |         |
| [`LearnAPI.predict_output_type(algorithm)`](@ref)   | dictionary of upper bounds on the type of predictions, keyed on subtypes of [`LearnAPI.KindOfProxy`](@ref)    |         |

## Implementation guide

A single-argument trait is declared following this pattern:

```julia
LearnAPI.is_pure_julia(algorithm::MyAlgorithmType) = true
```

A shorthand for single-argument traits is available:

```julia
@trait MyAlgorithmType is_pure_julia=true
```

Multiple traits can be declared like this:


```julia
@trait(
    MyAlgorithmType,
    is_pure_julia = true,
    pkg_name = "MyPackage",
)
```

### [The global trait contract](@id trait_contract)

To ensure that trait metadata can be stored in an external algorithm registry, LearnAPI.jl
requires:

1. *Finiteness:* The value of a trait is the same for all `algorithm`s with same value of
   [`LearnAPI.constructor(algorithm)`](@ref). This typically means trait values do not
   depend on type parameters! There is an exception if `is_composite(algorithm) = true`.

2. *Immediate serializability:* It should be possible to call a trait without first
   installing any third party package. Importing the package that defines the algorithm,
   together with `import LearnAPI` should suffice.

Because of 1, combining a lot of functionality into one algorithm (e.g. the algorithm can
perform both classification or regression) can mean traits are necessarily less
informative (as in `LearnAPI.predict_type(algorithm) = Any`).


## Reference

```@docs
LearnAPI.constructor
LearnAPI.functions
LearnAPI.kinds_of_proxy
LearnAPI.tags
LearnAPI.is_pure_julia
LearnAPI.pkg_name
LearnAPI.pkg_license
LearnAPI.doc_url
LearnAPI.load_path
LearnAPI.is_composite
LearnAPI.human_name
LearnAPI.data_interface
LearnAPI.iteration_parameter
LearnAPI.fit_scitype
LearnAPI.fit_type
LearnAPI.fit_observation_scitype
LearnAPI.fit_observation_type
LearnAPI.target_observation_scitype
LearnAPI.predict_input_scitype
LearnAPI.predict_input_observation_scitype
LearnAPI.predict_input_type
LearnAPI.predict_input_observation_type
LearnAPI.predict_output_scitype
LearnAPI.predict_output_type
LearnAPI.transform_input_scitype
LearnAPI.transform_input_observation_scitype
LearnAPI.transform_input_type
LearnAPI.transform_input_observation_type
LearnAPI.predict_or_transform_mutates
LearnAPI.transform_output_scitype
LearnAPI.transform_output_type
LearnAPI.@trait
```
