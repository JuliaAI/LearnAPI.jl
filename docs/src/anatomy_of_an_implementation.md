# Anatomy of an Implementation

> **Summary.** Formally, an **algorithm** is a container for the hyperparameters of some
> ML/statistics algorithm.  A basic implementation of the ridge regressor requires
> implementing `fit` and `predict` methods dispatched on the algorithm type; `predict` is
> an example of an **operation**, the others are `transform` and `inverse_transform`. In
> this example we also implement an **accessor function**, called `feature_importance`,
> returning the absolute values of the linear coefficients. The ridge regressor has a
> target variable and outputs literal predictions of the target (rather than, say,
> probabilistic predictions); accordingly the overloaded `predict` method is dispatched on
> the `TrueTarget` subtype of `KindOfProxy`. An **algorithm trait** declares this type as the
> preferred kind of target proxy. Other traits articulate the algorithm's training data type
> requirements and the input/output type of `predict`.

We begin by describing an implementation of LearnAPI.jl for basic ridge regression
(without intercept) to introduce the main actors in any implementation.


## Defining an algorithm type

The first line below imports the lightweight package LearnAPI.jl whose methods we will be
extending, the second, libraries needed for the core algorithm.

```@example anatomy
using LearnAPI
using LinearAlgebra, Tables
nothing # hide
```

Next, we define a struct to store the single hyperparameter `lambda` of this algorithm:

```@example anatomy
struct MyRidge <: LearnAPI.Algorithm
        lambda::Float64
end
nothing # hide
```

The subtyping `MyRidge <: LearnAPI.Algorithm` is optional but recommended where it is not
otherwise disruptive.

Instances of `MyRidge` are called **algorithms** and `MyRidge` is an **algorithm type**.

A keyword argument constructor providing defaults for all hyperparameters should be
provided:

```@example anatomy
nothing # hide
MyRidge(; lambda=0.1) = MyRidge(lambda)
nothing # hide
```

## Implementing training (fit)

A ridge regressor requires two types of data for training: **input features** `X` and a
[**target**](@ref scope) `y`. Training is implemented by overloading `fit`. Here `verbosity` is an integer
(`0` should train silently, unless warnings are needed):

```@example anatomy
function LearnAPI.fit(algorithm::MyRidge, verbosity, X, y)

        # process input:
        x = Tables.matrix(X)  # convert table to matrix
        s = Tables.schema(X)
        features = s.names

        # core solver:
        coefficients = (x'x + algorithm.lambda*I)\(x'y)

        # prepare output - learned parameters:
        fitted_params = (; coefficients)

        # prepare output - algorithm state:
        state = nothing  # not relevant here

        # prepare output - byproducts of training:
        feature_importances =
                [features[j] => abs(coefficients[j]) for j in eachindex(features)]
        sort!(feature_importances, by=last) |> reverse!
        verbosity > 0 && @info "Features in order of importance: $(first.(feature_importances))"
        report = (; feature_importances)

        return fitted_params, state, report
end
nothing # hide
```

Regarding the return value of `fit`:

- The `fitted_params` variable is for the algorithm's learned parameters, for passing to
  `predict` (see below).

- The `state` variable is only relevant when additionally implementing a [`LearnAPI.update!`](@ref)
  or [`LearnAPI.ingest!`](@ref) method (see [Fit, update! and ingest!](@ref)).

- The `report` is for other byproducts of training, apart from the learned parameters (the
  ones we'll need to provide `predict` below).

Our `fit` method assumes that `X` is a table (satisfies the [Tables.jl
spec](https://github.com/JuliaData/Tables.jl)) whose rows are the observations; and it
will need need `y` to be an `AbstractFloat` vector. An algorithm implementation is free to
dictate the representation of data that `fit` accepts but articulates its requirements
using appropriate traits; see [Training data types](@ref) below. We recommend against data
type checks internal to `fit`; this would ordinarily be the responsibility of a higher
level API, using those traits. 


## Operations

Now we need a method for predicting the target on new input features:

```@example anatomy
function LearnAPI.predict(::MyRidge, ::LearnAPI.TrueTarget, fitted_params, Xnew)
    Xmatrix = Tables.matrix(Xnew)
    report = nothing
    return Xmatrix*fitted_params.coefficients, report
end
nothing # hide
```

The second argument of `predict` is always an instance of `KindOfProxy`, and will always
be `TrueTarget()` in this case, as only literal values of the target (rather than, say
probabilistic predictions) are being supported.

In some algorithms `predict` computes something of interest in addition to the target
prediction, and this `report` item is returned as the second component of the return
value. When there's nothing to report, we must return `nothing`, as here.

Our `predict` method is an example of an **operation**. Other operations include
`transform` and `inverse_transform` and an algorithm can implement more than one. For
example, a K-means clustering algorithm might implement `transform` for dimension
reduction, and `predict` to return cluster labels. 

The `predict` method is reserved for predictions of a [target variable](@ref proxy), and
only `predict` has the extra `::KindOfProxy` argument.


## Accessor functions

The arguments of an operation are always `(algorithm, fitted_params, data...)`. The
interface also provides **accessor functions** for extracting information, from the
`fitted_params` and/or fit `report`, that is shared by several algorithm types.  There is
one for feature importances that we can implement for `MyRidge`:

```@example anatomy
LearnAPI.feature_importances(::MyRidge, fitted_params, report) =
    report.feature_importances
nothing # hide
```

Another example of an accessor function is [`LearnAPI.training_losses`](@ref).


## [Algorithm traits](@id traits)

We have implemented `predict`, and it is possible to implement `predict` methods for
multiple `KindOfProxy` types (see See [Target proxies](@ref) for a complete
list). Accordingly, we are required to declare a preferred target proxy, which we do using
[`LearnAPI.preferred_kind_of_proxy`](@ref):

```@example anatomy
LearnAPI.preferred_kind_of_proxy(::Type{<:MyRidge}) = LearnAPI.TrueTarget()
nothing # hide
```
Or, you can use the shorthand

```@example anatomy
@trait MyRidge preferred_kind_of_proxy=LearnAPI.TrueTarget()
nothing # hide
```

[`LearnAPI.preferred_kind_of_proxy`](@ref) is an example of a **algorithm trait**. A
complete list of traits and the contracts they imply is given in [Algorithm Traits](@ref).

We also need to indicate that a target variable appears in training (this is a supervised
algorithm). We do this by declaring *where* in the list of training data arguments (in this
case `(X, y)`) the target variable (in this case `y`) appears:

```@example anatomy
@trait MyRidge position_of_target=2
nothing # hide
```

As explained in the introduction, LearnAPI.jl does not attempt to define strict algorithm
categories, such as "regression" or "clustering". However, we can optionally specify suggestive
descriptors, as in

```@example anatomy
@trait MyRidge descriptors=(:regression,)
nothing # hide
```

This declaration actually promises nothing, but can help in generating documentation. Do
`LearnAPI.descriptors()` to get a list of available descriptors.

Finally, we are required to declare what methods (excluding traits) we have explicitly
overloaded for our type:

```@example anatomy
@trait MyRidge methods=(
        :fit,
        :predict,
        :feature_importances,
)
nothing # hide
```

## Training data types

Since LearnAPI.jl is a basement level API, one is discouraged from including explicit type
checks in an implementation of `fit`. Instead one uses traits to make promises about the
acceptable type of `data` consumed by `fit`. In general, this can be a promise regarding
the ordinary type of `data` or the [scientific
type](https://github.com/JuliaAI/ScientificTypes.jl) of `data` (but not
both). Alternatively, one may only promise a bound on the type/scitype of *observations*
in the data . See [Algorithm Traits](@ref) for further details. In this case we'll be
happy to restrict the scitype of the data:

```@example anatomy
import ScientificTypesBase: scitype, Table, Continuous
@trait MyRidge fit_scitype = Tuple{Table(Continuous), AbstractVector{Continuous}}
nothing # hide
```

This is a contract that `data` is acceptable in the call `fit(algorithm, verbosity, data...)`
whenever

```julia
scitype(data) <: Tuple{Table(Continuous), AbstractVector{Continuous}}
```

Or, in other words:

- `X` in `fit(algorithm, verbosity, X, y)` is acceptable, provided `scitype(X) <:
  Table(Continuous)` - meaning that `X` `Tables.istable(X) == true` (see
  [Tables.jl](https://github.com/JuliaData/Tables.jl)) and each column has some
  `<:AbstractFloat` element type.

- `y` in `fit(algorithm, verbosity, X, y)` is acceptable if `scitype(y) <:
  AbstractVector{Continuous}` - meaning that it is an abstract vector with `<:AbstractFloat`
  elements.

## Input types for operations

An optional promise about what `data` is guaranteed to work in a call like
`predict(algorithm, fitted_params, data...)` is articulated this way:

```@example anatomy
@trait MyRidge predict_input_scitype = Tuple{AbstractVector{<:Continuous}}
```

Note that `data` is always a `Tuple`, even if it has only one component (the typical
case), which explains the `Tuple` on the right-hand side.

Optionally, we may express our promise using regular types, using the
[`LearnAPI.predict_input_type`](@ref) trait.

One can optionally make promises about the outut of an operation. See [Algorithm
Traits](@ref) for details.


## [Illustrative fit/predict workflow](@id workflow)

We now illustrate how to interact directly with `MyRidge` instances using the methods we
have implemented:

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
Instantiate an algorithm with relevant hyperparameters (which is all the object stores):

```@example anatomy
algorithm = MyRidge(lambda=0.5)
```

Train the algorithm (the `0` means do so silently):

```@example anatomy
import LearnAPI: fit, predict, feature_importances

fitted_params, state, fit_report = fit(algorithm, 0, X[train], y[train])
```

Inspect the learned parameters and report:

```@example anatomy
@info "training outcomes" fitted_params fit_report
```

Inspect feature importances:

```@example anatomy
feature_importances(algorithm, fitted_params, fit_report)
```

Make a prediction using new data:

```@example anatomy
yhat, predict_report = predict(algorithm, LearnAPI.TrueTarget(), fitted_params, X[test])
```

Compare predictions with ground truth

```@example anatomy
deviations = yhat - y[test]
loss = deviations .^2 |> sum
@info "Sum of squares loss" loss
```
