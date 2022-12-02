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
`some_model.epochs`. Calling `Base.propertynames(some_model)` must return the names of
those hyper-parameters.

It is supposed that making copies of model objects is a cheap operation. Consequently,
*learned* parameters, such as weights in a neural network (the `fitted_params` described
in [Fit, update! and ingest!](@ref)) are not expected to be part of a model. Storing
learned parameters in a model is not explicitly ruled out, but doing so might lead to
performance issues in packages adopting LearnAPI.jl.

The only formal requirements of models are properties 1 and 2 given below in the
following explanation of the an **optional** supertype `LearnAPI.Model` for model
types:

```@docs
LearnAPI.Model
```

### Example

Any instance of `GradientRidgeRegressor` defined below is a valid LearnAPI.jl model:

```julia
struct GradientRidgeRegressor{T<:Real} <: LearnAPI.Model
    learning_rate::T
    epochs::Int
    l2_regularization::T
end
```

The same is true if we omit the subtyping `<: LearnAPI.Model`, but not if we also make
this a `mutable struct`. In that case we will need to overload `Base.==` for
`GradientRidgeRegressor`.

A keyword constructor providing default values for *all* non-model hyper-parameters is
required. If a model has other models as hyper-parameters, its
[`LearnAPI.is_wrapper`](@ref) trait must be set to `true`.


## Methods

None of the methods described in the linked sections below are compulsory, but any
implemented or overloaded method that is not a model trait must be added to the return
value of [`LearnAPI.methods`](@ref), as in

```julia
LearnAPI.methods(::Type{<SomeModelType}) = (:fit, update!, predict)
```

For examples, see [Anatomy of an Interface](@ref).

- [Fit, update! and ingest!](@ref): for models that "learn" (generalize to
  new data)

- [Operations](@ref operations): `predict`, `transform` and their relatives

- [Accessor Functions](@ref): accessing certain byproducts of training that many models
  share, such as feature importances and training losses

- [Model Traits](@ref): contracts for specific behaviour, such as "The second data
  argument of `fit` is a target variable" or "I predict probability distributions".
