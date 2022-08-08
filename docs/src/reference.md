# Reference

Here we give the definitive specification of the Learn API. For a more informal
guide see [Common Implementation Patterns](@ref).

## Models

> **Summary** In the Learn API a **model** is a Julia object whose properties are
> the hyper-parameters of some learning algorithm. Functionality is created by overloading
> methods defined by the interface and promises of certain behavior articulated by model
> traits.

In this document the word "model" has a very specific meaning that may differ from the
reader's common understanding of the word - in statistics, for example. In this document a
**model** is any julia object, `some_model` say, storing the hyper-parameters of some
learning algorithm that are accessible as named properties of the model, as in
`some_model.epochs`. Calling `Base.propertynames(some_model)` must return the names of those
hyper-parameters.

It is supposed that making copies of model objects is a cheap operation. Consequently,
*learned* parameters, such as coefficients in a linear model, or weights in a neural network
(the `fitted_params` appearing in [Fit, update! and ingest!](@ref)) are not expected to be
part of a model. Storing learned parameters in a model is not explicitly ruled out, but
doing so might lead to performance issues in packages adopting the Learn API.

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

and overload `Base.==` in the mutable case. 

> **MLJ only.** The subtyping also ensures instances will be displayed according to a
> standard MLJ convention, assuming MLJ or MLJBase are loaded.

```@docs
LearnAPI.ismodel
LearnAPI.Model
```

## Data containers

In this document a **data container** is any object implementing some kind of iteration
interface, where the length of the iteration, called the **number of observations**, is
known in advance.  At present "some kind of iteration interface" remains undefined, but a
working definition would include the `getrows` interface from
[Tables.jl](https://github.com/JuliaData/Tables.jl) interface and/or the `getobs` interface
from [MLUtils.jl](https://github.com/JuliaML/MLUtils.jl) (the latter interface
[subsuming](https://github.com/JuliaML/MLUtils.jl/issues/61) the former at some point?). The
`getobs` interface includes a built-in implementation for any `AbstractArray`, where the
observation index is understood to be the *last* index. Unfortunately, according to this
convention, a matrix `X` in this interface, corresponds to `Tables.table(X')` in the
`getrows` interface (where observations are rows).


## Methods

Model functionality is created by implementing:

- zero or more of the training methods, `fit`, `update!` and `ingest!` (the second and third
  require the first)

- zero or more **operations**, like `predict`

- zero or more **accessor functions**

While promises of certain behaviour are articulated using **model traits**. Examples of all
these methods given in [Anatomy of an Interface](@ref)).

- [Fit, update! and ingest!](@ref): for models that "learn" (generalize to
  new data)

- [Operations](@ref operations): `predict`, `transform` and their relatives

- [Accessor Functions](@ref): accessing byproducts of training shared by some models, such
  as feature importances and training losses

- [Model Traits](@ref): contracts for specific behaviour, such as "I am supervised" or "I
  predict probability distributions"
