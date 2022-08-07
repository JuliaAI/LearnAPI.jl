# Anatomy of an Implementation

> **Summary.** A **model** is just a container for hyper-parameters. A basic implementation
> for a ridge regressor requires implementing `fit` and `predict` methods dispatched on the
> model type; `predict` is an example of an **operation**; another is `transform`. In this
> example we also implement an **accessor function** called `feature_importance` (returning
> the absolute values of the linear coefficients). We need trait declarations to flag the
> model as supervised, and another to list the implemented methods. Optional traits
> articulate the model's data type requirements and the output type of operations.

We begin by describing an implementation of the ML Model Interface for basic ridge
regression (no intercept) to introduce the main actors in any implementation.


## Defining a model type

The first line below imports the lightweight package MLInterface.jl whose methods we will be
extending, the second libraries needed for the core algorithm.

```julia
import MLInterface
using LinearAlgebra, Tables
```

Next, we define a struct to store the single hyper-parameter `lambda` of this model:

```julia
struct MyRidge <: MLInterface.Model
	lambda::Float64
end
```

The subtyping `MyRidge <: MLInterface.Model` is optional but recommended where it is not
otherwise disruptive. If you omit the subtyping then you must declare

```julia
MLInterface.ismodel(::MyRidge) = true
```

as a promise that instances of `MyRidge` implement the compulsory elements of the ML Model
Interface.

Instances of `MyRidge` are called **models** and `MyRidge` is a **model type**.

A keyword argument constructor providing default hyper-parameters is strongly recommended:

```julia
MyRidge(; lambda=0.1) = MyRidge(lambda)
```

## A method to fit the model

A ridge regressor requires two types of data for training: **input features** `X` and a
**target** `y`. Training is implemented by overloading `fit`. Here `verbosity` is an integer
(`0` should train silently, unless warnings are needed):

```julia
function MLInterface.fit(model::MyRidge, verbosity, X, y)

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

- The `fitted_params` is for the model's learned parameters, in any form, for passing to
  `predict` (see below).

- The `state` variable is only relevant when additionally implementing an [`update`](@ref)
  or [`ingest`](@ref) method (see [Fit, update and ingest](@ref)).

- The `report` is for other byproducts of training, excluding the learned parameters.

Notice that we have chosen here to suppose that `X` is presented as a table (rows are the
observations); and we suppose `y` is a `Real` vector. (While this is typical of MLJ model
implementations, the ML Model Interface puts no restrictions on the form of `X` and `y`.)


## Operations

Now we need a method for predicting the target on new input features:

```julia
MLInterface.predict(::MyRidge, fitted_params, Xnew) = Tables.matrix(Xnew)*fitted_params.coefficients
```

The above `predict` method is an example of an **operation**. Other operations include
`transform` and `inverse_transform` and a model can implement more than one. For example, a
K-means clustering model might implement a `transform` for dimension reduction, and a
`predict` to return cluster labels.


## Accessor functions

The arguments of an operation are always `(model, fitted_params, data...)`. The interface also
provides **accessor functions** for extracting information from the `fitted_params` and/or
`report` that is shared by several model types.  There is one for feature importances that
we can implement for `MyRidge`:

```julia
MLInterface.feature_importances(::MyRidge, fitted_params, report) = report.feature_importances
```

Another example of an accessor function is `training_losses`.


## Model traits

In this supervised learning example, `predict` returns an object with the same type of the
*second* data argument `y` of `fit` (the target). It therefore makes sense, for example, to
apply a suitable metric (e.g., a sum of squares) to the pair `(ŷ, y)`, where `ŷ =
predict(model, fitted_params, X)`. We will flag this behavior by declaring

```julia
MLInterface.is_supervised(::Type{<:MyRidge}) = true
```

This is an example of a **model trait** declaration. A complete list of traits and the
contracts they imply is given in [`Model traits`](@ref).

> **MLJ only.** The values of all traits constitute a model's **metadata**, which is
> recorded in the searchable MLJ Model Registry, assuming the implementation-providing
> package is registered there.

Since our model is supervised, we are required to implement an additional trait that
distinguishes our model from other regressors that make probabilistic or other kinds of
predictions of the target:

```julia
MLInterface.prediction_type(::Type{<:MyRidge}) = :deterministic
```

As explained in the introduction, the ML Model Interface does not attempt to define strict
model "types", such as "regressor" or "clusterer". We can optionally specify
suggestive keywords, as in

```julia
MLJInterface.keywords(::Type{<:MyRidge}) = [:regression,]
```

but note that this declaration promises nothing. Do `MLInterface.keywords()` to get a list
of available keywords.

Finally, we are required to declare what methods (excluding traits) we have explicitly
overloaded for our type:

```julia
MLInterface.implemented_methods(::Type{<:MyRidge}) = [
	:fit,
	:predict,
	:feature_importances,
]
```

## Training data types

Optional trait declarations articulate the permitted types for training data. To be precise,
an implementation makes [scientific type](https://github.com/JuliaAI/ScientificTypes.jl)
declarations, which in this case look like:

```julia
using ScientificTypesBase
MLInterface.fit_data_scitype(::Type{<:MyRidge}) = Tuple{Table(Continuous), AbstractVector{Continuous}}
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


## Operation data types

A promise that an operation, such as `predict`, returns an object of given scientific type is articulated in this way:

```julia
MLJInterface.return_scitypes(::Type{<:MyRidge}) = Dict(:predict => AbstractVector{<:Continuous})
```

If `predict` had instead returned probability distributions, and these implement the
`Distributions.pdf` interface, then the declaration would be

```julia
MLJInterface.return_scitypes(::Type{<:MyRidge}) = Dict(:predict => AbstractVector{Density{<:Continuous}})
```

There is also an `input_scitypes` trait for operations. However, this falls back to the
scitype for the first argument of `fit`, as inferred from `fit_data_scitype` (see above). So
we need not overload it here.


## Convenience macros
