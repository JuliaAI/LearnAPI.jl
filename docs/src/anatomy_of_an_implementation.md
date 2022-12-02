# Anatomy of an Implementation

> **Summary.** A **model** is just a container for hyper-parameters. A basic
> implementation of the ridge regressor requires implementing `fit` and `predict` methods
> dispatched on the model type; `predict` is an example of an **operation** (another is
> `transform`). In this example we also implement an **accessor function**, called
> `feature_importance`, returning the absolute values of the linear coefficients. The
> ridge regressor has a target variable and one trait declaration flags the output of
> `predict` as being a [proxy](@ref scope) for the target. Other traits articulate the
> model's training data type requirements and the input/output type of `predict`.

We begin by describing an implementation of LearnAPI.jl for basic ridge
regression (no intercept) to introduce the main actors in any implementation.


## Defining a model type

The first line below imports the lightweight package LearnAPI.jl whose methods we will be
extending, the second, libraries needed for the core algorithm.

```@example anatomy
using LearnAPI
using LinearAlgebra, Tables
nothing # hide
```

Next, we define a struct to store the single hyper-parameter `lambda` of this model:

```@example anatomy
struct MyRidge <: LearnAPI.Model
        lambda::Float64
end
nothing # hide
```

The subtyping `MyRidge <: LearnAPI.Model` is optional but recommended where it is not
otherwise disruptive (it allows models to be displayed in a standard way, for example).

Instances of `MyRidge` are called **models** and `MyRidge` is a **model type**.

A keyword argument constructor providing defaults for all hyper-parameters should be
provided:

```@example anatomy
nothing # hide
MyRidge(; lambda=0.1) = MyRidge(lambda)
nothing # hide
```

## A method to fit the model

A ridge regressor requires two types of data for training: **input features** `X` and a
[**target**](@ref scope) `y`. Training is implemented by overloading `fit`. Here `verbosity` is an integer
(`0` should train silently, unless warnings are needed):

```@example anatomy
function LearnAPI.fit(model::MyRidge, verbosity, X, y)

        # process input:
        x = Tables.matrix(X)  # convert table to matrix
        features = Tables.columnnames(X)

        # core solver:
        coefficients = (x'x + model.lambda*I)\(x'y)

        # prepare output - learned parameters:
        fitted_params = (; coefficients)

        # prepare output - model state:
        state = nothing  # not relevant here

        # prepare output - byproducts of training:
        feature_importances =
                [features[j] => abs(coefficients[j]) for j in eachindex(features)]
        sort!(feature_importances, by=last) |> reverse!
        verbosity > 1 && @info "Features in order of importance: $(first.(feature_importances))"
        report = (; feature_importances)

        return fitted_params, state, report
end
nothing # hide
```

Regarding the return value of `fit`:

- The `fitted_params` variable is for the model's learned parameters, for passing to
  `predict` (see below).

- The `state` variable is only relevant when additionally implementing a [`LearnAPI.update!`](@ref)
  or [`LearnAPI.ingest!`](@ref) method (see [Fit, update! and ingest!](@ref)).

- The `report` is for other byproducts of training, apart from the learned parameters (the
  ones will need to provide `predict` below).

Our `fit` method assumes that `X` is a table (satifies the [Tables.jl
spec](https://github.com/JuliaData/Tables.jl)) whose rows are the observations; and it
will need need `y` to be an `AbstractFloat` vector. A model implementation is free to
dictate the representation of data that `fit` accepts but articulates its requirements
using appropriate traits; see [Training data types](@ref) below. We recommend against data
type checks internal to `fit`; this would ordinarily be the responsibility of a higher
level API, using those trasits. 


## Operations

Now we need a method for predicting the target on new input features:

```@example anatomy
function LearnAPI.predict(::MyRidge, fitted_params, Xnew)
    Xmatrix = Tables.matrix(Xnew)
    report = nothing
    return Xmatrix*fitted_params.coefficients, report
end
nothing # hide
```

In some models `predict` computes something of interest in addition to the target
prediction, and this `report` item is returned as the second component of the return
value. When there's nothing to report, we must return `nothing`, as here.

Our `predict` method is an example of an **operation**. Other operations include
`transform` and `inverse_transform` and a model can implement more than one. For example,
a K-means clustering model might implement a `transform` for dimension reduction, and a
`predict` to return cluster labels.


## Accessor functions

The arguments of an operation are always `(model, fitted_params, data...)`. The interface
also provides **accessor functions** for extracting information, from the `fitted_params`
and/or fit `report`, that is shared by several model types.  There is one for feature
importances that we can implement for `MyRidge`:

```@example anatomy
LearnAPI.feature_importances(::MyRidge, fitted_params, report) =
    report.feature_importances
nothing # hide
```

Another example of an accessor function is [`training_losses`](@ref).


## [Model traits](@id traits) 

Our model has a target variable, in the sense outlined in [Scope and undefined
notions](@ref scope), and `predict` returns an object with exactly the same form as the
target. We indicate this behaviour by declaring

```@example anatomy
LearnAPI.target_proxy(::Type{<:MyRidge}) = (; predict=LearnAPI.TrueTarget())
nothing # hide
```
Or, you can use the shorthand

```@example anatomy
@trait MyRidge target_proxy = (; predict=LearnAPI.TrueTarget())
nothing # hide
```

More generally, `predict` only returns a *proxy* for the target, such as probability
distributions, and we would make a different declaration here. See [Target proxies](@ref)
for details.

`LearnAPI.target_proxy` is an example of a **model trait**. A complete list of traits
and the contracts they imply is given in [Model Traits](@ref).

> **MLJ only.** The values of all traits constitute a model's **metadata**, which is
> recorded in the searchable MLJ Model Registry, assuming the implementation-providing
> package is registered there.

We also need to indicate that a target variable appears in training (this is a supervised
model). We do this by declaring *where* in the list of training data arguments (in this
case `(X, y)`) the target variable (in this case `y`) appears:

```@example anatomy
@trait MyRidge position_of_target = 2
nothing # hide
```

As explained in the introduction, LearnAPI.jl does not attempt to define strict model
"types", such as "regressor" or "clusterer". However, we can optionally specify suggestive
descriptors, as in

```@example anatomy
@trait MyRidge descriptors = (:regression,)
nothing # hide
```

but note that this declaration promises nothing. Do `LearnAPI.descriptors()` to get a list
of available descriptors.

Finally, we are required to declare what methods (excluding traits) we have explicitly
overloaded for our type:

```@example anatomy
@trait MyRidge methods = (
        :fit,
        :predict,
        :feature_importances,
)
nothing # hide
```

## Training data types

Since LearnAPI.jl is a basement level API, one is discouraged from including explicit type
checks in an implementation of `fit`. Instead one uses traits to make promisises about the
acceptable type of `data` consumed by `fit`. In general, this can be a promise regarding
the ordinary type of `data` and/or the [scientific
type](https://github.com/JuliaAI/ScientificTypes.jl) of `data`. Alternatively, one may
only make a promise about the type/scitype of *observations* in the data . See [Model
Traits](@ref) for further details. In this case we'll be happy to restrict the scitype of
the data:

```@example anatomy
import ScientificTypesBase: scitype, Table, Continuous
@trait MyRidge fit_data_scitype = Tuple{Table(Continuous), AbstractVector{Continuous}}
nothing # hide
```

This is a contract that `data` is acceptable in the call `fit(model, verbosity, data...)`
whenever

```@example anatomy
scitype(data) <: Tuple{Table(Continuous), AbstractVector{Continuous}}
nothing # hide
```

Or, in other words:

- `X` in `fit(model, verbosity, X, y)` is acceptable, provided `scitype(X) <:
  Table(Continuous)` - meaning that `X` `Tables.istable(X) == true` (see
  [Tables.jl](https://github.com/JuliaData/Tables.jl)) and each column has some
  `<:AbstractFloat` element type.

- `y` in `fit(model, verbosity, X, y)` is acceptable if `scitype(y) <:
  AbstractVector{Continuous}` - meaning that it is an abstract vector with `<:AbstractFloat`
  elements.

## Input/output types for operations

An optional promise that an operation, such as `predict`, returns an object of given
scientific type is articulated in this way:

```@example anatomy
@trait output_scitypes = (; predict=AbstractVector{<:Continuous})
nothing # hide
```

If `predict` had instead returned probability distributions that implement the
`Distributions.pdf` interface, then one could instead make the declaration

```julia
@trait MyRidge output_scitypes = (; predict=AbstractVector{Density{<:Continuous}})
```

Similarly, there exists a trait called [`output_type`](@ref) for making promises on the
ordinary type resturned by an operation.

Finally, we'll make a promise about what `data` is acceptable in a call like
`predict(model, fitted_params, data...)`. Note that `data` is always a `Tuple`, even if it
has only one component (the typical case).

```example anatomy
@trait MyRidge input_scitype = (; predict=Tuple{AbstractVector{<:Continuous}})
```

## [Illustrative fit/predict workflow](@id workflow)

Here's some toy data for supervised learning:

```@example anatomy
using Tables

n = 10          # number of training observations
train = 1:6
test = 7:10

a, b, c = rand(n), rand(n), rand(n)
X = (; a, b, c) |> Tables.rowtable
y = 2a - b + 3c + 0.05*rand(n)
nothing # hide
```
Instantiate a model with relevant hyperparameters:

```@example anatomy
model = MyRidge(lambda=0.5)
```

Train the model:

```@example anatomy
import LearnAPI: fit, predict, feature_importances

fitted_params, state, fit_report = fit(model, 1, X[train], y[train])
```

Inspect the learned paramters and report:

```@example anatomy
@info "training outcomes" fitted_params fit_report
```

Inspect feature importances:

```@example anatomy
feature_importances(model, fitted_params, fit_report)
```

Make a prediction using new data:

```@example anatomy
yhat, predict_report = predict(model, fitted_params, X[test])
```

Compare predictions with ground truth

```@example anatomy
deviations = yhat - y[test]
loss = deviations .^2 |> sum
@info "Sum of squares loss" loss
```
