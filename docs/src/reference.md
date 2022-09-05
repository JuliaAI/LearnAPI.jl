# Reference

Here we give the definitive specification of interface provided by LearnAPI.jl. For a more
informal guide see [Common Implementation Patterns](@ref).

## Models

> **Summary** In LearnAPI.jl a **model** is a Julia object whose properties are the
> hyper-parameters of some learning algorithm. Functionality is created by overloading
> methods defined by the interface and promises of certain behavior are articulated by
> model traits.

In this document the word "model" has a very specific meaning that may differ from the
reader's common understanding of the word - in statistics, for example. In this document a
**model** is any julia object, `some_model` say, storing the hyper-parameters of some
learning algorithm that are accessible as named properties of the model, as in
`some_model.epochs`. Calling `Base.propertynames(some_model)` must return the names of those
hyper-parameters.

It is supposed that making copies of model objects is a cheap operation. Consequently,
*learned* parameters, such as weights in a neural network (the `fitted_params` described
in [Fit, update! and ingest!](@ref)) are not expected to be part of a model. Storing
learned parameters in a model is not explicitly ruled out, but doing so might lead to
performance issues in packages adopting LearnAPI.jl.

Two models with the same type should be `==` if and only if all their hyper-parameters are
`==`. Of course, a hyper-parameter could be another model.

Any instance of `SomeType` below is a model in the above sense:

```julia
struct SomeType{T<:Real} <: LearnAPI.Model
    epochs::Int
    lambda::T
end
```

The subtyping `<: LearnAPI.Model` is optional. If it is included and the type is instead
a `mutable struct`, then there is no need to explicitly overload `Base.==`. If it is
omitted, then one must make the declaration

`LearnAPI.ismodel(::SomeType) = true`

and overload `Base.==` if the type is mutable.

> **MLJ only.** The subtyping also ensures instances will be displayed according to a
> standard MLJ convention, assuming MLJ or MLJBase is loaded.

```@docs
LearnAPI.ismodel
LearnAPI.Model
```

## Methods

None of the methods described in the linked sections below are compulsory, but any
implemented or overloaded method that is not a model trait must be added to the return
value of [`LearnAPI.implemented_methods`](@ref), as in

```julia
LearnAPI.implemented_methods(::Type{<SomeModelType}) = (:fit, update!, predict)
```

For examples, see [Anatomy of an Interface](@ref).

- [Fit, update! and ingest!](@ref): for models that "learn" (generalize to
  new data)

- [Operations](@ref operations): `predict`, `transform` and their relatives

- [Accessor Functions](@ref): accessing certain byproducts of training that many models
  share, such as feature importances and training losses

- [Model Traits](@ref): contracts for specific behaviour, such as "I am supervised" or "I
  predict probability distributions"
