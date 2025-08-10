# [Learner Traits](@id traits)

Learner traits are simply functions whose sole argument is a learner.

Traits promise specific learner behavior, such as: *This learner can make point or
probabilistic predictions* or *This learner is supervised* (sees a target in
training). They may also record more mundane information, such as a package license.

## [Trait summary](@id trait_summary)

### [Overloadable traits](@id traits_list)

In the examples column of the table below, `Continuous` is a name owned the package
[ScientificTypesBase.jl](https://github.com/JuliaAI/ScientificTypesBase.jl/).

| trait                                                    | return value                                                                                                           | fallback value                                       | example                                                        |
|:---------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------|:---------------------------------------------------------------|
| [`LearnAPI.constructor`](@ref)`(learner)`                | constructor for generating new or modified versions of `learner`                                                       | (no fallback)                                        | `RidgeRegressor`                                               |
| [`LearnAPI.functions`](@ref)`(learner)`                  | functions you can apply to `learner` or associated model (traits excluded)                                             | `()`                                                 | `(:fit, :predict, :LearnAPI.strip, :(LearnAPI.learner), :obs)` |
| [`LearnAPI.kind_of`](@ref)`(learner)`                    | the `fit`/`predict`/`transform` pattern used by `learner`                                                              | `LearnAPI.Static()`                                  | `LearnAPI.Descriminative()`                                    |
| [`LearnAPI.kinds_of_proxy`](@ref)`(learner)`             | instances `kind` of `KindOfProxy` for which an implementation of `LearnAPI.predict(learner, kind, ...)` is guaranteed. | `()`                                                 | `(Distribution(), Interval())`                                 |
| [`LearnAPI.tags`](@ref)`(learner)`                       | lists one or more suggestive learner tags from `LearnAPI.tags()`                                                       | `()`                                                 | (:regression, :probabilistic)                                  |
| [`LearnAPI.is_pure_julia`](@ref)`(learner)`              | `true` if implementation is 100% Julia code                                                                            | `false`                                              | `true`                                                         |
| [`LearnAPI.pkg_name`](@ref)`(learner)`                   | name of package providing core code (may be different from package providing LearnAPI.jl implementation)               | `"unknown"`                                          | `"DecisionTree"`                                               |
| [`LearnAPI.pkg_license`](@ref)`(learner)`                | name of license of package providing core code                                                                         | `"unknown"`                                          | `"MIT"`                                                        |
| [`LearnAPI.doc_url`](@ref)`(learner)`                    | url providing documentation of the core code                                                                           | `"unknown"`                                          | `"https://en.wikipedia.org/wiki/Decision_tree_learning"`       |
| [`LearnAPI.load_path`](@ref)`(learner)`                  | string locating name returned by `LearnAPI.constructor(learner)`, beginning with a package name                        | `"unknown"`                                          | `FastTrees.LearnAPI.DecisionTreeClassifier`                    |
| [`LearnAPI.nonlearners`](@ref)`(learner)`                | properties *not* corresponding to other learners                                                                       | all properties                                       | `(:K, :leafsize, :metric,)`                                    |
| [`LearnAPI.human_name`](@ref)`(learner)`                 | human name for the learner; should be a noun                                                                           | type name with spaces                                | "elastic net regressor"                                        |
| [`LearnAPI.iteration_parameter`](@ref)`(learner)`        | symbolic name of an iteration parameter                                                                                | `nothing`                                            | :epochs                                                        |
| [`LearnAPI.data_interface`](@ref)`(learner)`             | Interface implemented by objects returned by [`obs`](@ref)                                                             | `Base.HasLength()` (supports `MLCore.getobs/numobs`) | `Base.SizeUnknown()` (supports `iterate`)                      |
| [`LearnAPI.fit_scitype`](@ref)`(learner)`                | upper bound on `scitype(data)` ensuring `fit(learner, data)` works                                                     | `Union{}`                                            | `Tuple{AbstractVector{Continuous}, Continuous}`                |
| [`LearnAPI.target_observation_scitype`](@ref)`(learner)` | upper bound on the scitype of each observation of the targget                                                          | `Any`                                                | `Continuous`                                                   |
| [`LearnAPI.is_static`](@ref)`(learner)`                  | `true` if `fit` consumes no data                                                                                       | `false`                                              | `true`                                                         |

### Derived Traits

The following are provided for convenience but should not be overloaded by new learners:

| trait                          | return value                                                             | example       |
|:-------------------------------|:-------------------------------------------------------------------------|:--------------|
| `LearnAPI.name(learner)`       | learner type name as string                                              | "PCA"         |
| `LearnAPI.learners(learner)`   | properties with learner values                                           | `(:atom, )` |
| `LearnAPI.is_learner(learner)` | `true` if `learner` is LearnAPI.jl-compliant                             | `true`        |
| `LearnAPI.target(learner)`     | `true` if `fit` sees a target variable; see [`LearnAPI.target`](@ref)    | `false`       |
| `LearnAPI.weights(learner)`    | `true` if `fit` supports per-observation; see [`LearnAPI.weights`](@ref) | `false`       |

## Implementation guide

Only `LearnAPI.constructor` and `LearnAPI.functions` are universally compulsory. 

A single-argument trait is declared following this pattern:

```julia
LearnAPI.is_pure_julia(learner::MyLearnerType) = true
```

A macro [`@trait`](@ref) provides a short-cut:

```julia
@trait MyLearnerType is_pure_julia=true
```

Multiple traits can be declared like this:


```julia
@trait(
    MyLearnerType,
    is_pure_julia = true,
    pkg_name = "MyPackage",
)
```

### [The global trait contract](@id trait_contract)

To ensure that trait metadata can be stored in an external learner registry, LearnAPI.jl
requires:

1. *Finiteness:* The value of a trait is the same for all `learner`s with same value of
   [`LearnAPI.constructor(learner)`](@ref). This typically means trait values do not
   depend on type parameters! For composite models (non-empty
   `LearnAPI.learners(learner)`) this requirement is dropped.

2. *Low level deserializability:* It should be possible to evaluate the trait *value* when
   `LearnAPI` and `ScientificTypesBase` are the only imported modules. 

Because of 1, combining a lot of functionality into one learner (e.g. the learner can
perform both classification or regression) can mean traits are necessarily less
informative (as in `LearnAPI.target_observation_scitype(learner) = Any`).


## Reference

```@docs
LearnAPI.constructor
LearnAPI.functions
LearnAPI.kind_of
LearnAPI.kinds_of_proxy
LearnAPI.tags
LearnAPI.is_pure_julia
LearnAPI.pkg_name
LearnAPI.pkg_license
LearnAPI.doc_url
LearnAPI.load_path
LearnAPI.nonlearners
LearnAPI.human_name
LearnAPI.data_interface
LearnAPI.iteration_parameter
LearnAPI.fit_scitype
LearnAPI.target_observation_scitype
LearnAPI.is_static
```

```@docs
LearnAPI.learners
```
