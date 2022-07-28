# MLInterface.jl

An Julia interface for training and applying models in machine learning and statistics


&#x1F6A7;

| Linux | Coverage |
| :------------ | :------- |
| [![Build Status](https://github.com/JuliaAI/MLInterface.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/MLInterface.jl/actions) | [![Coverage](https://codecov.io/gh/JuliaAI/MLInterface.jl/branch/master/graph/badge.svg)](https://codecov.io/github/JuliaAI/MLInterface.jl?branch=master) |

**Status.** Proposal stage (no code)

This repository is to provide a general purpose machine learning interface. It is designed
based on experiences of developers of MLJ's [MLJModelInterface.jl]() which it will
eventually replace, but hopes to be useful more generally. The design is in a state of flux
and comments (posted as issues) are welcome.

The interface makes wide use of traits to articulate model functionality. There is no
abstract model type heirarchy. Model data type requirements can be articulated using
[scientific types](https://github.com/JuliaAI/ScientificTypes.jl) but this is optional.

**Proposal.** The proposal (WIP) is in the form of documentation that lives in this
README.md for now:

---

Machine learning algorithms, sometimes called *models*, have a complicated taxonomy. In our
experience, grouping algorithms into a relatively small number of types, such as
"classifier" and "clusterer", and attempting to impose uniform behaviour within each group,
is overly limiting. Accordingly, the behaviour of a model implementing the **ML Model
Interface** documented here is articulated using traits - methods dispatched on the model
type, such as `is_supervised(model::SomeModel) = true` and
`prediction_type(model::SomeModel) = :probabilistic`.

Although designed to support [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/)'s
"machine" interface for user interaction, the interface described here is a general purpose,
standalone API for machine learning algorithms, implemented by extending methods in
MLInterface.jl, which is lightweight (and has no reference to machines).

The preceding observations notwithstanding, a new implementation of the ML Model Interface
will often fall into one of the [Common Implementation Patterns](@ref) described first. The
definitive specification of the interface is provided in the [Reference](@ref) section.

- [Anatomy of an Implementation](@ref)

- [Common Implementation Patterns](@ref)

- [Testing an implementation](@ref)

- [Reference](@ref)



# Anatomy of an Implementation

> **Summmary.** A **model** is just a container for hyper-parameters. A basic implementation
> for a ridge regressor requires implementing `fit` and `predict` methods dispatched on the
> model type; `predict` is an example of an **operation**; another is `transform`. In this
> exaple we can also implement an **accessor function** called `feature_importance`
> (returning the absolute values of the linear coefficients). Finally, we need trait
> declarations to flag the model as supervised, and that its predictions as deterministic
> (non-probabilistic). Optional traits articulate the model's data type requirements.

!!! important

    This introductory section introduces terminology essential in the sequel.

We begin by describing an implementation of the ML Model Interface for a naive
zero-intercept ridge regression algorithm (training is one line of Julia) to introduce the
main actors in any implementation.

The first line below imports the lightweight package MLInterface.jl whose methods we
will be extending:

```julia
import MLInterface
import Tables
using LinearAlgebra
```

Next, we define a struct to store the single hyper-parameter `lambda` of this model:

```julia
struct MyRidge
    lambda::Float64
end
```

Instances of `MyRidge` are called **models** and `MyRidge` is a **model type**.

> **MLJ Only.** To ensure that `MyRidge` instances are displayed using MLJ's standard, add the
> optional subtyping `MyRidge <: MLInterface.Model`.

A keyword argument constructor providing default hyper-parameters is strongly recommended:

```julia
MyRidge(; lambda=0.1) = MyRidge(lambda)
```

A ridge regressor requires two types of data for training: **input features** `X` and a
**target** `y`. Training is implemented by overloading `fit`; here `verbosity` is an integer
(`0` should train silently, unless warnings are needed):

```julia
function MLInterface.fit(model::MyRidge, verbosity, X, y)

    # process input:
    x = Tables.matrix(X)  # convert table to matrix
    features = Tables.columnnames(X)

    # core solver:
    coefficients = (x'x + model.lambda*I)\(x'y)

    # prepare output - learned parameters:
    fitresult = (; coefficients)

    # prepare output - model state:
    state = nothing  # only relevant for models also implementing an `update` method

    # prepare output - byproducts of training:
    feature_importances =
        [features[j] => abs(coefficients[j]) for j in eachindex(features)]
    sort!(feature_importances, by=last) |> reverse!
    verbosity > 1 && @info "Features in order of importance: $(first.(feature_importances))"
    report = (; feature_importances)

    return fitresult, state, report
end
```

The `fitresult` is for the model's learned parameters. The `state` variable is only relevant
when additionally implementing an [`update`](@ref) method, which is for updating learned
parameters after a change in hyper-parameters (e.g., an increase in a iteration parameter)
or because of additional data (incremental learning). The `report` is for other byproducts
of training in which a user may be interested in.

Notice that we have chosen here to suppose that `X` is presented as a table (rows are the
observations); we suppose `y` is a `Real` vector. This is typical of MLJ implementations.

Now we need a method for predicting the target on new input features:

```julia
MLInterface.predict(::MyRidge, fitresult, Xnew) = Tables.matrix(Xnew)*fitresult.coefficients
```

The above `predict` method is an example of an **operation**. Other operations include
`transform` and `inverse_transform` and a model can implement more than one. For example, a
K-means clustering model might implement a `transform` for dimension reduction, and a
`predict` to return cluster labels.

The arguments of an operation are always `(model, fitresult, data...)`. The interface also
provides more specialized **accessor functions** which are called on `fitresult` and
`report`.  There is one that applies in this case, `feature_importance`:

```julia
MLInterface.feature_importances(::MyRidge, fitresult, report) = report.feature_importances
```

Now the data argument `Xnew` of `predict` has the same type as the *first* argument `X`
encountered in `fit`, while `predict` returns an object with the type of the *second* data
argument `y` of `fit`. It therefore makes sense, for example, to apply a suitable metric
(e.g., a sum of squares) to the pair `(ŷ, y)`, where `ŷ = predict(model, fitresult, X)`. We
will flag this behavior by declaring

```julia
MLInterface.is_supervised(::Type{<:MyRidge}) = true
```

This is an example of a **model trait** declaration. A complete list of traits and the
contracts they imply is given in TODO.

> **MLJ only.** The values of all traits constitute a model's **metadata**, which is
> recorded in the searchable MLJ Model Registry, assuming the implementation-providing
> package is registered there.

We also add a trait declaration to distinguish our ridge regressor from other regressors
that make *probabilistic* or *interval* predictions:

```julia
MLInterface.prediction_type(::Type{<:MyRidge}) = :deterministic
```

Such a declaration is required by any model implementing a `predict` method.

Finally, we are required to declare what methods (excluding traits) we have explicitly
overloaded for our type:

```julia
MLInterface.implemented_methods(::Type{<:MyRidge}) = [
    :fit, 
    :predict,
    :feature_importances,
]
```

## Articulating data type requirements

Optional trait declarations articulate the permitted types for training data. To be precise,
an implementation declares [scientific type](https://github.com/JuliaAI/ScientificTypes.jl),
which in this case would look like:

```julia
using ScientificTypesBase
fit_data_scitype(::Type{<:MyRidge}) = Tuple{Table(Continuous), AbstractVector{Continuous}}
```

This is a contract that `data` is acceptable input to ` in `fit(model, verbosity, data...)`
whenever

```julia
scitype(data) <: Tuple{Table(Continuous), AbstractVector{Continuous}}
```

Or in other words:

- `X` in `fit(model, verbosity, X, y)` is acceptable, provided `scitype(X) <:
Table(Continuous)` - meaning that `X` is a Tables.jl compatible table whose columns have
some `<:AbstractFloat` element type (and the same must be true `Xnew`
in `predict(model, fitresult, Xnew)`).

- `y` in `fit(model, verbosity, X, y)` is acceptable if `scitype(y) <:
AbstractVector{Continuous}` - meaning that it is an abstract vector with `<:AbstractFloat`
elements.

> **MLJ only.** In MLJ these types are used to assist in model search (matching models to
> data) and to issue informative warnings to users attempting to use invalid data.


## Convenience macros


# Common Implementation Patterns

!!! warning

    This section is not a definitive specification of the ML Model Interface.
        See instead [Reference](@ref).


# Reference

This section is a detailed specification of the ML Model Interface. For a more
informal discussion see [Common Implementation Patterns](@ref).

Strongly recommended prepartion is the section [Anatomy of a Model Implementation](@ref).


## Models

> **Summary** In the ML Model Interface a **model** is a Julia object whose properties are
> the hyper-parameters of some learning algorithm. The behaviour of a model is determined
> purely by the methods in MLInterface.jl that are overloaded for it.

In this document the word "model" has a very specific meaning that may conflict with the reader's
common understanding of the word - in statistics, for example. In this document a **model** is
any julia object `some_model` storing the hyper-parameters of some learning algorithm that
are accessible as named properties of the model, as in `some_model.epochs`. Calling
`Base.propertynames(some_model)` must return the names of those hyper-parameters.

Two models with the same type should be `==` if and only if all their hyper-parameters are
`==`. Of course, a hyper-parameter could be another model.

Any instance of `SomeModel` below is a model in the above sense:

```julia
struct SomeModel{T<:Real} <: MLInterface.Model
        epochs::Int
        lambda::T
end
```

The subtyping `MLInterface.Model <: Model` is optional. If it is included and the type is
instead a `mutable struct`, then there is no need to explicitly overload `Base.==`.

> **MLJ only.** The subtyping also ensures instances will be displayed according to a
> standard MLJ convention, assuming MLJ or MLJBase are loaded.

To be useful, models must be wrapped in some functionality, which is achieved by overloading
methods defined in MLInterface.jl, as described in the following sections below. The ML
Model Interface does not define an abstract model type heirarchy; rather one guarantees
additional behaviour by implementing appropriate traits summarized in TODO.


## The `fit` method



## Summary of traits
