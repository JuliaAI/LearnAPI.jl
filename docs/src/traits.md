# [Algorithm Traits](@id traits)

Algorithm traits are simply functions whose sole argument is an algorithm.

Traits promise specific algorithm behavior, such as: *This algorithm can make point or
probabilistic predictions* or *This algorithm is supervised* (sees a target in
training). They may also record more mundane information, such as a package license.

## [Trait summary](@id trait_summary)

### [Overloadable traits](@id traits_list)

In the examples column of the table below, `Continuous` is a name owned the package
[ScientificTypesBase.jl](https://github.com/JuliaAI/ScientificTypesBase.jl/).

| trait                                                        | return value                                                                                                             | fallback value                                        | example                                                    |
|:-------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------|:-----------------------------------------------------------|
| [`LearnAPI.constructor`](@ref)`(algorithm)`                  | constructor for generating new or modified versions of `algorithm`                                                       | (no fallback)                                         | `RidgeRegressor`                                           |
| [`LearnAPI.functions`](@ref)`(algorithm)`                    | functions you can apply to `algorithm` or associated model (traits excluded)                                             | `()`                                                  | `(:fit, :predict, :minimize, :(LearnAPI.algorithm), :obs)` |
| [`LearnAPI.kinds_of_proxy`](@ref)`(algorithm)`               | instances `kind` of `KindOfProxy` for which an implementation of `LearnAPI.predict(algorithm, kind, ...)` is guaranteed. | `()`                                                  | `(Distribution(), Interval())`                             |
| [`LearnAPI.tags`](@ref)`(algorithm)`                         | lists one or more suggestive algorithm tags from `LearnAPI.tags()`                                                       | `()`                                                  | (:regression, :probabilistic)                              |
| [`LearnAPI.is_pure_julia`](@ref)`(algorithm)`                | `true` if implementation is 100% Julia code                                                                              | `false`                                               | `true`                                                     |
| [`LearnAPI.pkg_name`](@ref)`(algorithm)`                     | name of package providing core code (may be different from package providing LearnAPI.jl implementation)                 | `"unknown"`                                           | `"DecisionTree"`                                           |
| [`LearnAPI.pkg_license`](@ref)`(algorithm)`                  | name of license of package providing core code                                                                           | `"unknown"`                                           | `"MIT"`                                                    |
| [`LearnAPI.doc_url`](@ref)`(algorithm)`                      | url providing documentation of the core code                                                                             | `"unknown"`                                           | `"https://en.wikipedia.org/wiki/Decision_tree_learning"`   |
| [`LearnAPI.load_path`](@ref)`(algorithm)`                    | string locating name returned by `LearnAPI.constructor(algorithm)`, beginning with a package name                        | "unknown"`                                            | `FastTrees.LearnAPI.DecisionTreeClassifier`                |
| [`LearnAPI.is_composite`](@ref)`(algorithm)`                 | `true` if one or more properties of `algorithm` may be an algorithm                                                      | `false`                                               | `true`                                                     |
| [`LearnAPI.human_name`](@ref)`(algorithm)`                   | human name for the algorithm; should be a noun                                                                           | type name with spaces                                 | "elastic net regressor"                                    |
| [`LearnAPI.iteration_parameter`](@ref)`(algorithm)`          | symbolic name of an iteration parameter                                                                                  | `nothing`                                             | :epochs                                                    |
| [`LearnAPI.data_interface`](@ref)`(algorithm)`               | Interface implemented by objects returned by [`obs`](@ref)                                                               | `Base.HasLength()` (supports `MLUtils.getobs/numobs`) | `Base.SizeUnknown()` (supports `iterate`)                  |
| [`LearnAPI.fit_observation_scitype`](@ref)`(algorithm)`      | upper bound on `scitype(observation)` for `observation` in `data` ensuring `fit(algorithm, data)` works                  | `Union{}`                                             | `Tuple{AbstractVector{Continuous}, Continuous}`            |
| [`LearnAPI.target_observation_scitype`](@ref)`(algorithm)`   | upper bound on the scitype of each observation of the targget                                                            | `Any`                                                 | `Continuous`                                               |
| [`LearnAPI.predict_or_transform_mutates`](@ref)`(algorithm)` | `true` if `predict` or `transform` mutates first argument                                                                | `false`                                               | `true`                                                     |

### Derived Traits

The following are provided for convenience but should not be overloaded by new algorithms:

| trait                              | return value                                                             | example |
|:-----------------------------------|:-------------------------------------------------------------------------|:--------|
| `LearnAPI.name(algorithm)`         | algorithm type name as string                                            | "PCA"   |
| `LearnAPI.is_algorithm(algorithm)` | `true` if `algorithm` is LearnAPI.jl-compliant                           | `true`  |
| `LearnAPI.target(algorithm)`       | `true` if `fit` sees a target variable; see [`LearnAPI.target`](@ref)    | `false` |
| `LearnAPI.weights(algorithm)`      | `true` if `fit` supports per-observation; see [`LearnAPI.weights`](@ref) | `false` |

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
   depend on type parameters! If `is_composite(algorithm) = true`, this requirement is
   dropped.

2. *Low level deserializability:* It should be possible to evaluate the trait *value* when
   `LearnAPI` is the only imported module. 

Because of 1, combining a lot of functionality into one algorithm (e.g. the algorithm can
perform both classification or regression) can mean traits are necessarily less
informative (as in `LearnAPI.target_observation_scitype(algorithm) = Any`).


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
LearnAPI.fit_observation_scitype
LearnAPI.target_observation_scitype
LearnAPI.predict_or_transform_mutates
```
