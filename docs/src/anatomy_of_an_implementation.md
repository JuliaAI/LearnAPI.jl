# Anatomy of an Implementation

> **Summary.** A **model** is just a container for hyper-parameters. A basic
> implementation of the ridge regressor requires implementing `fit` and `predict` methods
> dispatched on the model type; `predict` is an example of an **operation** (another is
> `transform`). In this example we also implement an **accessor function** called
> `feature_importance` (returning the absolute values of the linear coefficients). The
> ridge regressor has a target variable and one trait declaration flags the output of
> `predict` as being a [proxy](@ref scope) for the target. Other traits articulate the
> model's training data type requirements and the output type of `predict`.

We begin by describing an implementation of LearnAPI.jl for basic ridge
regression (no intercept) to introduce the main actors in any implementation.


## Defining a model type

The first line below imports the lightweight package LearnAPI.jl whose methods we will be
extending, the second, libraries needed for the core algorithm.

```julia
using LearnAPI
using LinearAlgebra, Tables
```

Next, we define a struct to store the single hyper-parameter `lambda` of this model:

```julia
struct MyRidge <: LearnAPI.Model
        lambda::Float64
end
```

The subtyping `MyRidge <: LearnAPI.Model` is optional but recommended where it is not
otherwise disruptive. If you omit the subtyping then you must declare

```julia
LearnAPI.ismodel(::MyRidge) = true
```

as a promise that instances of `MyRidge` implement LearnAPI.jl.

Instances of `MyRidge` are called **models** and `MyRidge` is a **model type**.

A keyword argument constructor providing default hyper-parameters is strongly recommended:

```julia
MyRidge(; lambda=0.1) = MyRidge(lambda)
```

## A method to fit the model

A ridge regressor requires two types of data for training: **input features** `X` and a
[**target**](@ref scope) `y`. Training is implemented by overloading `fit`. Here `verbosity` is an integer
(`0` should train silently, unless warnings are needed):

```julia
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
```

Regarding the return value of `fit`:

- The `fitted_params` variable is for the model's learned parameters, for passing to
  `predict` (see below).

- The `state` variable is only relevant when additionally implementing a [`LearnAPI.update!`](@ref)
  or [`LearnAPI.ingest!`](@ref) method (see [Fit, update! and ingest!](@ref)).

- The `report` is for other byproducts of training, excluding the learned parameters.

Notice that we have chosen here to suppose that `X` is presented as a table (rows are the
observations); and we suppose `y` is a `Real` vector. This is not a restriction on types
placed by LearnAPI.jl. However, we can articulate our model's particular type requirements
with the [`LearnAPI.fit_data_scitype`](@ref) trait; see [Training data types](@ref) below.


## Operations

Now we need a method for predicting the target on new input features:

```julia
function LearnAPI.predict(::MyRidge, fitted_params, Xnew)
    Xmatrix = Tables.matrix(Xnew)
    report = nothing
    return Xmatrix*fitted_params.coefficients, report
end
```

In some models `predict` computes something of interest in addition to the target
prediction, and this `report` item is returned as the second component of the return
value. When there's nothing to report, we must return `nothing`, as here.

Our `predict` method is an example of an **operation**. Other operations include
`transform` and `inverse_transform` and a model can implement more than one. For example,
a K-means clustering model might implement a `transform` for dimension reduction, and a
`predict` to return cluster labels.


## Accessor functions

The arguments of an operation are always `(model, fitted_params, data...)`. The interface also
provides **accessor functions** for extracting information, from the `fitted_params` and/or
`report`, that is shared by several model types.  There is one for feature importances that
we can implement for `MyRidge`:

```julia
LearnAPI.feature_importances(::MyRidge, fitted_params, report) =
report.feature_importances
```

Another example of an accessor function is `training_losses`.


## [Model traits](@id traits) 

Our model has a target variable, in the sense outlined in [Scope and undefined
notions](@ref scope), and `predict` returns an object with exactly the same form as the
target. We indicate this behaviour by declaring

```julia
LearnAPI.target_proxy_kind(::Type{<:MyRidge}) = (; predict=LearnAPI.TrueTarget())
```
Or, you can use the shorthand

```julia
@trait MyRidge target_proxy_kind = (; predict=LearnAPI.TrueTarget())
```

More generally, `predict` only returns a *proxy* for the target, such as probability
distributions, and we would make a different declaration here. See [Target proxies](@ref)
for details.

`LearnAPI.target_proxy_kind` is an example of a **model trait**. A complete list of traits
and the contracts they imply is given in [Model Traits](@ref).

> **MLJ only.** The values of all traits constitute a model's **metadata**, which is
> recorded in the searchable MLJ Model Registry, assuming the implementation-providing
> package is registered there.

We also need to indicate that the target appears in training (this is a *supervised*
model) and the position of `target` within the `data` argument of `fit`:

```julia
@trait MyRidge position_of_target = 2
```

As explained in the introduction, LearnAPI.jl does not attempt to define strict model
"types", such as "regressor" or "clusterer". However, we can optionally specify suggestive
keywords, as in

```julia
@trait MyRidge keywords = (:regression,)
```

but note that this declaration promises nothing. Do `LearnAPI.keywords()` to get a list
of available keywords.

Finally, we are required to declare what methods (excluding traits) we have explicitly
overloaded for our type:

```julia
@trait MyRidge implemented_methods = (
        :fit,
        :predict,
        :feature_importances,
)
```

## Training data types

Optional trait declarations articulate the permitted types for training data. To be precise,
an implementation makes [scientific type](https://github.com/JuliaAI/ScientificTypes.jl)
declarations, which in this case look like:

```julia
using ScientificTypesBase
@trait MyRidge fit_data_scitype = Tuple{Table(Continuous), AbstractVector{Continuous}}
```

This is a contract that `data` is acceptable in the call `fit(model, verbosity, data...)`
whenever

```julia
scitype(data) <: Tuple{Table(Continuous), AbstractVector{Continuous}}
```

Or, in other words:

- `X` in `fit(model, verbosity, X, y)` is acceptable, provided `scitype(X) <:
  Table(Continuous)` - meaning that `X` is a Tables.jl compatible table whose columns have
  some `<:AbstractFloat` element type.

- `y` in `fit(model, verbosity, X, y)` is acceptable if `scitype(y) <:
  AbstractVector{Continuous}` - meaning that it is an abstract vector with `<:AbstractFloat`
  elements.


## Types for data returned by operations

A promise that an operation, such as `predict`, returns an object of given scientific type
is articulated in this way:

```julia
@trait return_scitypes = (:predict => AbstractVector{<:Continuous},)
```

If `predict` had instead returned probability distributions, and these implement the
`Distributions.pdf` interface, then the declaration would be

```julia
@trait return_scitypes = (:predict => AbstractVector{Density{<:Continuous}},)
```

There is also an `input_scitypes` trait for operations. However, this falls back to the
scitype for the first argument of `fit`, as inferred from `fit_data_scitype` (see above). So
we need not overload it here.


## [Illustrative fit/predict workflow](@id workflow)

Here's some toy data for supervised learning:

```julia
using Tables

n = 10          # number of training observations
train = 1:6
test = 7:10

a, b, c = rand(n), rand(n), rand(n)
X = (; a, b, c) |> Tables.rowtable
y = 2a - b + 3c + 0.05*rand(n)
```
Instantiate a model with relevant hyperparameters:

```julia
model = MyRidge(lambda=0.5)
```

Train the model:

```julia
import LearnAPI: fit, predict, feature_importances

fitted_params, state, fit_report = fit(model, 1, X[train], y[train])
```

Inspect the learned paramters and report:

```julia
@info "training outcomes" fitted_params report
```

Inspect feature importances:

```julia
feature_importances(model, fitted_params, report)
```

Make a prediction using new data:

```julia
yhat, predict_report = predict(model, fitted_params, X[test])
```

Compare predictions with ground truth

```julia
deviations = yhat - y[test]
loss = deviations .^2 |> sum
@info "Sum of squares loss" loss
```
