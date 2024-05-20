# Anatomy of an Implementation

This section explains a detailed implementation of the LearnAPI for naive [ridge
regression](https://en.wikipedia.org/wiki/Ridge_regression) with no intercept. The kind of
workflow we want to enable has been previewed in [Sample workflow](@ref). Readers can also
refer to the [demonstration](@ref workflow) of the implementation given later.

For a transformer, implementations ordinarily implement `transform` instead of
`predict`. For more on `predict` versus `transform`, see [Predict or transform?](@ref)

!!! important

	The core implementations of `fit`, `predict`, etc,
	always have a *single* `data` argument, as in `fit(algorithm, data; verbosity=1)`.
	Calls like `fit(algorithm, X, y)` are provided as additional convenience methods.

!!! note

	If the `data` object consumed by `fit`, `predict`, or `transform` is not
	not a suitable table¹, array³, tuple of tables and arrays, or some
	other object implementing
	the MLUtils.jl `getobs`/`numobs` interface,
	then an implementation must: (i) suitably overload the trait
	[`LearnAPI.data_interface`](@ref); and/or (ii) overload [`obs`](@ref), as
	illustrated below under [Providing an advanced data interface](@ref).

The first line below imports the lightweight package LearnAPI.jl whose methods we will be
extending. The second imports libraries needed for the core algorithm.

```@example anatomy
using LearnAPI
using LinearAlgebra, Tables
nothing # hide
```

## Defining algorithms

Here's a new type whose instances specify ridge regression parameters:

```@example anatomy
struct Ridge{T<:Real}
		lambda::T
end
nothing # hide
```

Instances of `Ridge` will be [algorithms](@ref algorithms), in LearnAPI.jl parlance.

To [qualify](@ref algorithms) as a LearnAPI algorithm, an object must be come with a
mechanism for creating new versions of itself, with modified property (field) values. To
this end, we implement `LearnAPI.constructor`, which must return a keyword constructor:

```@example anatomy
Ridge(; lambda=0.1) = Ridge(lambda)
LearnAPI.constructor(::Ridge) = Ridge
nothing # hide
```

So, if `algorithm = Ridge(lambda=0.1)` then `LearnAPI.constructor(algorithm)(lambda=0.05)`
is another algorithm with the same properties, except that the value of `lambda` has been
changed to `0.05`.


## Implementing `fit`

A ridge regressor requires two types of data for training: *input features* `X`, which
here we suppose are tabular¹, and a [target](@ref proxy) `y`, which we suppose is a
vector.

It is convenient to define a new type for the `fit` output, which will include
coefficients labelled by feature name for inspection after training:

```@example anatomy
struct RidgeFitted{T,F}
		algorithm::Ridge
		coefficients::Vector{T}
		named_coefficients::F
end
nothing # hide
```

Note that we also include `algorithm` in the struct, for it must be possible to recover
`algorithm` from the output of `fit`; see [Accessor functions](@ref) below.

The core implementation of `fit` looks like this:

```@example anatomy
function LearnAPI.fit(algorithm::Ridge, data; verbosity=1)

		X, y = data

		# data preprocessing:
		table = Tables.columntable(X)
		names = Tables.columnnames(table) |> collect
		A = Tables.matrix(table, transpose=true)

		lambda = algorithm.lambda

		# apply core algorithm:
		coefficients = (A*A' + algorithm.lambda*I)\(A*y) # vector

		# determine named coefficients:
		named_coefficients = [names[j] => coefficients[j] for j in eachindex(names)]

		# make some noise, if allowed:
		verbosity > 0 && @info "Coefficients: $named_coefficients"

		return RidgeFitted(algorithm, coefficients, named_coefficients)
end
```

## Implementing `predict`

The primary `predict` call will look like this:

```julia
predict(model, LiteralTarget(), Xnew)
```

where `Xnew` is a table (of the same form as `X` above). The argument `LiteralTarget()`
signals that we want literal predictions of the target variable, as opposed to a proxy for
the target, such as probability density functions.  `LiteralTarget` is an example of a
[`LearnAPI.KindOfProxy`](@ref proxy_types) type. Targets and target proxies are discussed
[here](@ref proxy).

Here's the implementation for our ridge regressor:

```@example anatomy
LearnAPI.predict(model::RidgeFitted, ::LiteralTarget, Xnew) =
		Tables.matrix(Xnew)*model.coefficients
```

## Accessor functions

An [accessor function](@ref accessor_functions) has the output of [`fit`](@ref) as it's
sole argument.  Every new implementation must implement the accessor function
[`LearnAPI.algorithm`](@ref) for recovering an algorithm from a fitted object:

```@example anatomy
LearnAPI.algorithm(model::RidgeFitted) = model.algorithm
```

Other accessor functions extract learned parameters or some standard byproducts of
training, such as feature importances or training losses.² Here we implement an accessor
function to extract the linear coefficients:

```@example anatomy
LearnAPI.coefficients(model::RidgeFitted) = model.named_coefficients
nothing #hide
```

## Tearing a model down for serialization

The `minimize` method falls back to the identity. Here, for the sake of illustration, we
overload it to dump the named version of the coefficients:

```@example anatomy
LearnAPI.minimize(model::RidgeFitted) =
		RidgeFitted(model.algorithm, model.coefficients, nothing)
```

Crucially, we can still use `LearnAPI.minimize(model)` in place of `model` to make new
predictions.


## Algorithm traits

Algorithm [traits](@ref traits) record extra generic information about an algorithm, or
make specific promises of behavior. They usually have an algorithm as the single argument,
and so we also regard [`LearnAPI.constructor`](@ref) defined above as a trait.

In LearnAPI.jl `predict` always outputs a [target or target proxy](@ref proxy), where
"target" is understood very broadly. We overload a trait to record the fact here that the
target variable explicitly appears in training (i.e, the algorithm is supervised):

```julia
LearnAPI.target(::Ridge) = true
```

or, using a shortcut:

```julia
@trait Ridge target = true
```

The macro can be used to specify multiple traits simultaneously:

```@example anatomy
@trait(
	Ridge,
	constructor = Ridge,
	target = true,
	kinds_of_proxy=(LiteralTarget(),),
	descriptors = (:regression,),
	functions = (
		fit,
		minimize,
		predict,
		obs,
		LearnAPI.algorithm,
		LearnAPI.coefficients,
	)
)
nothing # hide
```

The trait `kinds_of_proxy` is required here, because we implemented `predict`.

The last trait `functions` returns a list of all LearnAPI.jl methods that can be
meaninfully applied to the algorithm or associated model. See [`LearnAPI.functions`](@ref)
for a checklist.  This, and [`LearnAPI.constructor`](@ref), are the only universally
compulsory traits. However, it is worthwhile studying the [list of all traits](@ref
traits_list) to see which might apply to a new implementation, to enable maximum buy into
functionality provided by third party packages, and to assist third party algorithms that
match machine learning algorithms to user-defined tasks.

According to the contract articulated in its document string, having set
[`LearnAPI.target(::Ridge)`](@ref) equal to `true`, we are obliged to overload a
multi-argument version of `LearnAPI.target` to extract the target from the `data` that
gets supplied to `fit`:

```@example anatomy
LearnAPI.target(::Ridge, data) = last(data)
```

## Convenience methods

Finally, we extend `fit` and `predict` with signatures convenient for user interaction,
enabling the kind of workflow previewed in [Sample workflow](@ref):

```@example anatomy
LearnAPI.fit(algorithm::Ridge, X, y; kwargs...) =
	fit(algorithm, (X, y); kwargs...)

LearnAPI.predict(model::RidgeFitted, Xnew) =
	predict(model, LiteralTarget(), Xnew)
```

## [Demonstration](@id workflow)

We now illustrate how to interact directly with `Ridge` instances using the methods
just implemented.

```@example anatomy
# synthesize some data:
n = 10 # number of observations
train = 1:6
test = 7:10
a, b, c = rand(n), rand(n), rand(n)
X = (; a, b, c)
y = 2a - b + 3c + 0.05*rand(n)
nothing # hide
```

```@example anatomy
algorithm = Ridge(lambda=0.5)
foreach(println, LearnAPI.functions(algorithm))
```

Training and predicting:

```@example anatomy
model = fit(algorithm, Tables.subset(X, train), y[train])
ŷ = predict(model, LiteralTarget(), Tables.subset(X, test))
```

Extracting coefficients:

```@example anatomy
LearnAPI.coefficients(model)
```

Serialization/deserialization:

```@example anatomy
using Serialization
small_model = minimize(model)
filename = tempname()
serialize(filename, small_model)
```

```julia
recovered_model = deserialize(filename)
@assert LearnAPI.algorithm(recovered_model) == algorithm
@assert predict(recovered_model, X) == predict(model, X)
```

## Providing an advanced data interface

```@setup anatomy2
using LearnAPI
using LinearAlgebra, Tables

struct Ridge{T<:Real}
		lambda::T
end

Ridge(; lambda=0.1) = Ridge(lambda)

struct RidgeFitted{T,F}
		algorithm::Ridge
		coefficients::Vector{T}
		named_coefficients::F
end

LearnAPI.algorithm(model::RidgeFitted) = model.algorithm
LearnAPI.coefficients(model::RidgeFitted) = model.named_coefficients
LearnAPI.minimize(model::RidgeFitted) =
		RidgeFitted(model.algorithm, model.coefficients, nothing)

LearnAPI.fit(algorithm::Ridge, X, y; kwargs...) =
	fit(algorithm, (X, y); kwargs...)
LearnAPI.predict(model::RidgeFitted, Xnew) = predict(model, LiteralTarget(), Xnew)

@trait(
	Ridge,
	constructor = Ridge,
	target = true,
	kinds_of_proxy=(LiteralTarget(),),
	descriptors = (:regression,),
	functions = (
		fit,
		minimize,
		predict,
		obs,
		LearnAPI.algorithm,
		LearnAPI.coefficients,
	)
)

n = 10 # number of observations
train = 1:6
test = 7:10
a, b, c = rand(n), rand(n), rand(n)
X = (; a, b, c)
y = 2a - b + 3c + 0.05*rand(n)
```

An implementation may optionally implement [`obs`](@ref), to expose to the user (or some
meta-algorithm like cross-validation) the representation of input data internal to `fit`
or `predict`, such as the matrix version `A` of `X` in the ridge example.  Here we
specifically wrap all the pre-processed data into single object, for which we introduce a
new type:

```@example anatomy2
struct RidgeFitObs{T,M<:AbstractMatrix{T}}
		A::M                  # p x n
		names::Vector{Symbol} # features
		y::Vector{T}          # target
end
```

Now we overload `obs` to carry out the data pre-processing previously in `fit`, like this:

```@example anatomy2
function LearnAPI.obs(::Ridge, data)
		X, y = data
		table = Tables.columntable(X)
		names = Tables.columnnames(table) |> collect
		return RidgeFitObs(Tables.matrix(table)', names, y)
end
```

We informally refer to the output of `obs` as "observations" (see [The `obs`
contract](@ref) below). The previous core `fit` signature is now replaced with two
methods - one to handle "regular" input, and one to handle the pre-processed data
(observations) which appears first below:

```@example anatomy2
function LearnAPI.fit(algorithm::Ridge, observations::RidgeFitObs; verbosity=1)

		lambda = algorithm.lambda

		A = observations.A
		names = observations.names
		y = observations.y

		# apply core algorithm:
		coefficients = (A*A' + algorithm.lambda*I)\(A*y) # 1 x p matrix

		# determine named coefficients:
		named_coefficients = [names[j] => coefficients[j] for j in eachindex(names)]

		# make some noise, if allowed:
		verbosity > 0 && @info "Coefficients: $named_coefficients"

		return RidgeFitted(algorithm, coefficients, named_coefficients)

end

LearnAPI.fit(algorithm::Ridge, data; kwargs...) =
		fit(algorithm, obs(algorithm, data); kwargs...)
```

We provide an overloading of `LearnAPI.target` to handle the additional supported data
argument of `fit`:

```@example anatomy2
LearnAPI.target(::Ridge, observations::RidgeFitObs) = observations.y
```

### The `obs` contract

Providing `fit` signatures matching the output of `obs`, is the first part of the `obs`
contract. The second part is this: *The output of `obs` must implement the*
[MLUtils.jl](https://juliaml.github.io/MLUtils.jl/dev/) `getobs/numobs` *interface for
accessing individual observations*. It usually suffices to overload `Base.getindex` and
`Base.length` (which are the `getobs/numobs` fallbacks):

```@example anatomy2
Base.getindex(data::RidgeFitObs, I) =
		RidgeFitObs(data.A[:,I], data.names, y[I])
Base.length(data::RidgeFitObs, I) = length(data.y)
```

We can do something similar for `predict`, but there's no need for a new type in this
case:

```@example anatomy2
LearnAPI.obs(::RidgeFitted, Xnew) = Tables.matrix(Xnew)'

LearnAPI.predict(model::RidgeFitted, ::LiteralTarget, observations::AbstractMatrix) =
		observations'*model.coefficients

LearnAPI.predict(model::RidgeFitted, ::LiteralTarget, Xnew) =
		predict(model, LiteralTarget(), obs(model, Xnew))
```

### Important notes:

- The observations to be consumed by `fit` are returned by `obs(algorithm::Ridge, ...)`,
  while those consumed by `predict` are returned by `obs(model::RidgeFitted, ...)`. We
  need the different signatures because the form of data consumed by `fit` and `predict`
  are generally different.

- We need the adjoint operator, `'`, because the last dimension in arrays is the
  observation dimension, according to the MLUtils.jl convention. Remember, `Xnew` is a
  table here.

Since LearnAPI.jl provides fallbacks for `obs` that simply return the unadulterated data
input, overloading `obs` is optional. This is provided data in publicized `fit`/`predict`
signatures consists of objects implementing the `getobs/numobs` interface (such as tables¹
and arrays³).

To buy out of supporting the MLUtils.jl interface altogether, an implementation must
overload the trait, [`LearnAPI.data_interface(algorithm)`](@ref).

For more on data interfaces, see [`obs`](@ref) and
[`LearnAPI.data_interface(algorithm)`](@ref).


## Demonstration of an advanced `obs` workflow

We now can train and predict using internal data representations, resampled using the
generic MLUtils.jl interface:

```@example anatomy2
import MLUtils
algorithm = Ridge()
observations_for_fit = obs(algorithm, (X, y))
model = fit(algorithm, MLUtils.getobs(observations_for_fit, train))
observations_for_predict = obs(model, X)
ẑ = predict(model, MLUtils.getobs(observations_for_predict, test))
```

```julia
@assert ẑ == ŷ
```

---

¹ In LearnAPI.jl a *table* is any object `X` implementing the
[Tables.jl](https://tables.juliadata.org/dev/) interface, additionally satisfying
`Tables.istable(X) == true` and implementing `DataAPI.nrow` (and whence
`MLUtils.numobs`). Tables that are also (unnamed) tuples are disallowed.

² An implementation can provide further accessor functions, if necessary, but
like the native ones, they must be included in the [`LearnAPI.functions`](@ref)
declaration.

³ The last index must be the observation index.

⁴ Guaranteed assuming
`LearnAPI.data_interface(algorithm) == Base.HasLength()`, the default.
