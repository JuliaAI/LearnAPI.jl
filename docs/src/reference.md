# [Reference](@id reference)

> **Summary** In LearnAPI.jl a **model** is a container for hyper-parameters of some
> learning algorithm. Functionality is created by overloading methods defined by the
> interface and promises of certain behavior are articulated by model traits.

Here we give the definitive specification of the interface provided by LearnAPI.jl. For a
more informal guide see [Common Implementation Patterns](@ref).

## Models

In this document the word "model" has a very specific meaning that may differ from the
reader's common understanding of the word - in statistics, for example.

Here a **model** is some julia object storing the hyper-parameters of some learning
algorithm. Typically the type of `m` will have a name reflecting that of the algormithm,
such as `DecisionTreeRegressor`.

Additionally, for `m::M` to be a LearnAPI model, we require:

- `Base.propertynames(m)` returns the hyper-parameters of `m`.

- If `m` is a model, then so are all instances of the same type.

- If `n` is another model, then `m == n` if and only if `typeof(n) == typeof(m)` and
  corresponding properties are `==`. This includes properties that are random number
  generators (which should be copied in training to avoid mutation).

- A keyword constructor for `M` exists, providing default values for *all* non-model
  hyper-parameters.

- If a model has other models as hyper-parameters, then [`LearnAPI.is_wrapper`](@ref)`(m)`
  must be `true`.

Whenever any LearnAPI method (excluding traits) is overloaded for some type `M` (e.g.,
`predict`, `transform`, `fit`) then that is a promise that all instances of `M` are
models. (In particular, [`LearnAPI.functions`](@ref)`(M)` will be non-empty in this case.)

It is supposed that making copies of model objects is a cheap operation. Consequently,
*learned* parameters, such as weights in a neural network (the `fitted_params` described
in [Fit, update! and ingest!](@ref)) are not expected to be part of a model. Storing
learned parameters in a model is not explicitly ruled out, but doing so might lead to
performance issues in packages adopting LearnAPI.jl.

A **model type** is a type whose instances are models.

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
`SomeModel`.

```@docs
LearnAPI.Model
```

## Methods

None of the methods described in the linked sections below are compulsory, but any
implemented or overloaded method that is not a model trait must be added to the return
value of [`LearnAPI.functions`](@ref), as in

```julia
LearnAPI.functions(::Type{<SomeModelType}) = (:fit, :update!, :predict)
```

or using the shorthand

```julia
@trait SomeModelType functions=(:fit, :update!, :predict)
```

For examples, see [Anatomy of an Implementation](@ref).

- [Fit, update! and ingest!](@ref): for models that "learn" (generalize to
  new data)

- [Operations](@ref operations): `predict`, `transform` and their relatives

- [Accessor Functions](@ref): accessing certain byproducts of training that many models
  share, such as feature importances and training losses

- [Model Traits](@ref): contracts for specific behaviour, such as "The second data
  argument of `fit` is a target variable" or "I predict probability distributions".
