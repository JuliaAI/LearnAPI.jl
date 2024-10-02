# Anatomy of an Implementation

This section explains a detailed implementation of the LearnAPI for naive [ridge
regression](https://en.wikipedia.org/wiki/Ridge_regression) with no intercept. The kind of
workflow we want to enable has been previewed in [Sample workflow](@ref). Readers can also
refer to the [demonstration](@ref workflow) of the implementation given later.

A transformer ordinarily implements `transform` instead of
`predict`. For more on `predict` versus `transform`, see [Predict or transform?](@ref)

!!! note

    New implementations of `fit`, `predict`, etc,
    always have a *single* `data` argument, as in
    `LearnAPI.fit(algorithm, data; verbosity=1) = ...`.
    For convenience, user-calls, such as `fit(algorithm, X, y)`, automatically fallback
    to `fit(algorithm, (X, y))`.

!!! note

    If the `data` object consumed by `fit`, `predict`, or `transform` is not
    not a suitable table¹, array³, tuple of tables and arrays, or some
    other object implementing
    the MLUtils.jl `getobs`/`numobs` interface,
    then an implementation must: (i) overload [`obs`](@ref) to articulate how
    provided data can be transformed into a form that does support
    this interface, as illustrated below under 
	[Providing an advanced data interface](@ref); or (ii) overload the trait
    [`LearnAPI.data_interface`](@ref) to specify a more relaxed data
    API. 

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

Associated with each new type of LearnAPI [algorithm](@ref algorithms) will be a keyword
argument constructor, providing default values for all properties (struct fields) that are
not other algorithms, and we must implement [`LearnAPI.constructor(algorithm)`](@ref), for
recovering the constructor from an instance:

```@example anatomy
"""
    Ridge(; lambda=0.1)

Instantiate a ridge regression algorithm, with regularization of `lambda`.
"""
Ridge(; lambda=0.1) = Ridge(lambda)
LearnAPI.constructor(::Ridge) = Ridge
nothing # hide
```

For example, in this case, if `algorithm = Ridge(0.2)`, then
`LearnAPI.constructor(algorithm)(lambda=0.2) == algorithm` is true. Note that we attach
the docstring to the *constructor*, not the struct.


## Implementing `fit`

A ridge regressor requires two types of data for training: input features `X`, which here
we suppose are tabular¹, and a [target](@ref proxy) `y`, which we suppose is a vector.

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

Users will be able to call `predict` like this:

```julia
predict(model, Point(), Xnew)
```

where `Xnew` is a table (of the same form as `X` above). The argument `Point()`
signals that literal predictions of the target variable are sought, as opposed to some
proxy for the target, such as probability density functions.  `Point` is an
example of a [`LearnAPI.KindOfProxy`](@ref proxy_types) type. Targets and target proxies
are discussed [here](@ref proxy).

We provide this implementation for our ridge regressor:

```@example anatomy
LearnAPI.predict(model::RidgeFitted, ::Point, Xnew) =
    Tables.matrix(Xnew)*model.coefficients
```

If the kind of proxy is omitted, as in `predict(model, Xnew)`, then a fallback grabs the
first element of the tuple returned by [`LearnAPI.kinds_of_proxy(algorithm)`](@ref), which
we overload appropriately below.


## Extracting the target from training data

The `fit` method consumes data which includes a [target variable](@ref proxy), i.e., the
algorithm is a supervised learner. We must therefore declare how the target variable can be extracted
from training data, by implementing [`LearnAPI.target`](@ref):

```@example anatomy
LearnAPI.target(algorithm, data) = last(data)
```

There is a similar method, [`LearnAPI.features`](@ref) for declaring how training features
can be extracted (for passing to `predict`, for example) but this method has a fallback
which typically suffices: return `first(data)` if `data` is a tuple, and otherwise return
`data`.


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
and so we regard [`LearnAPI.constructor`](@ref) defined above as a trait.

Because we have implemented `predict`, we are required to overload the
[`LearnAPI.kinds_of_proxy`](@ref) trait. Because we can only make point predictions of the
target, we make this definition:

```julia
LearnAPI.kinds_of_proxy(::Ridge) = (Point(),)
```

A macro provides a shortcut, convenient when multiple traits are to be defined:

```@example anatomy
@trait(
    Ridge,
    constructor = Ridge,
    kinds_of_proxy=(Point(),),
    tags = (:regression,),
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.algorithm),
        :(LearnAPI.minimize),
        :(LearnAPI.obs),
        :(LearnAPI.features),
        :(LearnAPI.target),
        :(LearnAPI.predict),
        :(LearnAPI.coefficients),
   )
)
nothing # hide
```

The last trait, `functions`, returns a list of all LearnAPI.jl methods that can be
meaninfully applied to the algorithm or associated model. See [`LearnAPI.functions`](@ref)
for a checklist.  [`LearnAPI.functions`](@ref) and [`LearnAPI.constructor`](@ref), are the
only universally compulsory traits. However, it is worthwhile studying the [list of all
traits](@ref traits_list) to see which might apply to a new implementation, to enable
maximum buy into functionality provided by third party packages, and to assist third party
algorithms that match machine learning algorithms to user-defined tasks.

Note that we know `Ridge` instances are supervised algorithms because `:(LearnAPI.target)
in LearnAPI.functions(algorithm)`, for every instance `algorithm`. With [some
exceptions](@ref trait_contract), the value of a trait should depend only on the *type* of
the argument.


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
Xtrain = Tables.subset(X, train)
ytrain = y[train]
model = fit(algorithm, (Xtrain, ytrain))  # `fit(algorithm, Xtrain, ytrain)` will also work
ŷ = predict(model, Tables.subset(X, test))
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

@trait(
    Ridge,
    constructor = Ridge,
    kinds_of_proxy=(Point(),),
    tags = (:regression,),
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.algorithm),
        :(LearnAPI.minimize),
        :(LearnAPI.obs),
        :(LearnAPI.features),
        :(LearnAPI.target),
        :(LearnAPI.predict),
        :(LearnAPI.coefficients),
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
    A::M                  # `p` x `n` matrix
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

### The `obs` contract

Providing `fit` signatures matching the output of `obs`, is the first part of the `obs`
contract. The second part is this: *The output of `obs` must implement the interface
specified by the trait* [`LearnAPI.data_interface(algorithm)`](@ref). Assuming this is
[`LearnAPI.RandomAccess()`](@ref) (the default) it usually suffices to overload
`Base.getindex` and `Base.length`:

```@example anatomy2
Base.getindex(data::RidgeFitObs, I) =
    RidgeFitObs(data.A[:,I], data.names, y[I])
Base.length(data::RidgeFitObs, I) = length(data.y)
```

We can do something similar for `predict`, but there's no need for a new type in this
case:

```@example anatomy2
LearnAPI.obs(::RidgeFitted, Xnew) = Tables.matrix(Xnew)'

LearnAPI.predict(model::RidgeFitted, ::Point, observations::AbstractMatrix) =
    observations'*model.coefficients

LearnAPI.predict(model::RidgeFitted, ::Point, Xnew) =
    predict(model, Point(), obs(model, Xnew))
```

### `target` and `features` methods

We provide an additional overloading of [`LearnAPI.target`](@ref) to handle the additional
supported data argument of `fit`:

```@example anatomy2
LearnAPI.target(::Ridge, observations::RidgeFitObs) = observations.y
```

Similarly, we must overload [`LearnAPI.features`](@ref), which extracts features from
training data (objects that can be passed to `predict`) like this

```@example anatomy2
LearnAPI.features(::Ridge, observations::RidgeFitObs) = observations.A
```
as the fallback mentioned above is no longer adequate.


### Important notes:

- The observations to be consumed by `fit` are returned by `obs(algorithm::Ridge, ...)`,
  while those consumed by `predict` are returned by `obs(model::RidgeFitted, ...)`. We
  need the different signatures because the form of data consumed by `fit` and `predict`
  are generally different.

- We need the adjoint operator, `'`, because the last dimension in arrays is the
  observation dimension, according to the MLUtils.jl convention. Remember, `Xnew` is a
  table here.

Since LearnAPI.jl provides fallbacks for `obs` that simply return the unadulterated data
argument, overloading `obs` is optional. This is provided data in publicized
`fit`/`predict` signatures consists only of objects implement the
[`LearnAPI.RandomAccess`](@ref) interface (most tables¹, arrays³, and tuples thereof).

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

For an application of [`obs`](@ref) to efficient cross-validation, see [here](@ref
obs_workflows).

---

¹ In LearnAPI.jl a *table* is any object `X` implementing the
[Tables.jl](https://tables.juliadata.org/dev/) interface, additionally satisfying
`Tables.istable(X) == true` and implementing `DataAPI.nrow` (and whence
`MLUtils.numobs`). Tables that are also (unnamed) tuples are disallowed.

² An implementation can provide further accessor functions, if necessary, but
like the native ones, they must be included in the [`LearnAPI.functions`](@ref)
declaration.

³ The last index must be the observation index.
