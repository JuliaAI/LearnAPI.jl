# Reference

Here we give the definitive specification of the ML Model Interface. For a more informal
guide see [Common Implementation Patterns](@ref).


## Models

> **Summary** In the ML Model Interface a **model** is a Julia object whose properties are
> the hyper-parameters of some learning algorithm. Functionality is created by overloading
> methods defined by the interface and promises of certain behavior articulated by model
> traits.

In this document the word "model" has a very specific meaning that may conflict with the
reader's common understanding of the word - in statistics, for example. In this document a
**model** is any julia object, `some_model` say, storing the hyper-parameters of some
learning algorithm that are accessible as named properties of the model, as in
`some_model.epochs`. Calling `Base.propertynames(some_model)` must return the names of those
hyper-parameters.

Two models with the same type should be `==` if and only if all their hyper-parameters are
`==`. Of course, a hyper-parameter could be another model.

Any instance of `SomeType` below is a model in the above sense:

```julia
struct SomeType{T<:Real} <: MLInterface.Model
    epochs::Int
    lambda::T
end
```

The subtyping `<: MLInterface.Model` is optional. If it is included and the type is instead
a `mutable struct`, then there is no need to explicitly overload `Base.==`. If it is
omitted, then one must make the declaration

`MLInterface.ismodel(::SomeType) = true`

and overload `Base.==` in the mutable case. 

> **MLJ only.** The subtyping also ensures instances will be displayed according to a
> standard MLJ convention, assuming MLJ or MLJBase are loaded.

```@docs
MLInterface.ismodel
MLInterface.Model
```

## Methods

Model functionality is created by implementing `fit` (and optionally `update!` and
`ingest!`), one or more *operations*, like `predict`, and optional **accessor functions**;
promises of certain behavior is articulated using **model traits**. Examples of these
methods are given in [Anatomy of an Interface](@ref)).

- [Fit, update! and ingest!](@ref): for models that "learn" (generalize to
  new data)

- [Operations](@ref): `predict`, `transform` and their relatives

- [Accessor Functions](@ref): accessing byproducts of training shared by some models, such
  as feature importances and training losses

- [Model Traits](@ref): contracts for specific behaviour, such as "I am supervised" or "I
  predict probability distributions"
