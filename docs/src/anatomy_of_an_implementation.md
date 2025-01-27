# Anatomy of an Implementation

The core LearnAPI.jl pattern looks like this:

```julia
model = fit(learner, data)
predict(model, newdata)
```

Here `learner` specifies hyperparameters, while `model` stores learned parameters and any byproducts of algorithm execution.

[Transformers](@ref) ordinarily implement `transform` instead of `predict`. For more on
`predict` versus `transform`, see [Predict or transform?](@ref)

["Static" algorithms](@ref static_algorithms) have a `fit` that consumes no `data`
(instead `predict` or `transform` does the heavy lifting). In [density
estimation](@ref density_estimation), `predict` consumes no data.

These are the basic possibilities.

Elaborating on the core pattern above, we detail in this tutorial an implementation of the
LearnAPI.jl for naive [ridge regression](https://en.wikipedia.org/wiki/Ridge_regression)
with no intercept. The kind of workflow we want to enable has been previewed in [Sample
workflow](@ref). Readers can also refer to the [demonstration](@ref workflow) of the
implementation given later.

!!! note

    New implementations of `fit`, `predict`, etc,
    always have a *single* `data` argument as above.
    For convenience, a signature such as `fit(learner, X, y)`, calling
    `fit(learner, (X, y))`, can be added, but the LearnAPI.jl specification is
    silent on the meaning or existence of signatures with extra arguments.

!!! note

    If the `data` object consumed by `fit`, `predict`, or `transform` is not
    not a suitable table¹, array³, tuple of tables and arrays, or some
    other object implementing
    the [MLUtils.jl](https://juliaml.github.io/MLUtils.jl/dev/) 
	`getobs`/`numobs` interface,
    then an implementation must: (i) overload [`obs`](@ref) to articulate how
    provided data can be transformed into a form that does support
    this interface, as illustrated below under
    [Providing a separate data front end](@ref); or (ii) overload the trait
    [`LearnAPI.data_interface`](@ref) to specify a more relaxed data
    API.

The first line below imports the lightweight package LearnAPI.jl whose methods we will be
extending. The second imports libraries needed for the core algorithm.

```@example anatomy
using LearnAPI
using LinearAlgebra, Tables
nothing # hide
```

## Defining learners

Here's a new type whose instances specify the single ridge regression hyperparameter:

```@example anatomy
struct Ridge{T<:Real}
    lambda::T
end
nothing # hide
```

Instances of `Ridge` are *[learners](@ref learners)*, in LearnAPI.jl parlance.

Associated with each new type of LearnAPI.jl learner will be a keyword
argument constructor, providing default values for all properties (typically, struct
fields) that are not other learners, and we must implement
[`LearnAPI.constructor(learner)`](@ref), for recovering the constructor from an instance:

```@example anatomy
"""
    Ridge(; lambda=0.1)

Instantiate a ridge regression learner, with regularization of `lambda`.
"""
Ridge(; lambda=0.1) = Ridge(lambda)
LearnAPI.constructor(::Ridge) = Ridge
nothing # hide
```

For example, in this case, if `learner = Ridge(0.2)`, then
`LearnAPI.constructor(learner)(lambda=0.2) == learner` is true. Note that we attach
the docstring to the *constructor*, not the struct.


## Implementing `fit`

A ridge regressor requires two types of data for training: input features `X`, which here
we suppose are tabular¹, and a [target](@ref proxy) `y`, which we suppose is a vector.⁴

It is convenient to define a new type for the `fit` output, which will include
coefficients labelled by feature name for inspection after training:

```@example anatomy
struct RidgeFitted{T,F}
    learner::Ridge
    coefficients::Vector{T}
    named_coefficients::F
end
nothing # hide
```

Note that we also include `learner` in the struct, for it must be possible to recover
`learner` from the output of `fit`; see [Accessor functions](@ref) below.

The implementation of `fit` looks like this:

```@example anatomy
function LearnAPI.fit(learner::Ridge, data; verbosity=LearnAPI.default_verbosity())

    X, y = data

    # data preprocessing:
    table = Tables.columntable(X)
    names = Tables.columnnames(table) |> collect
    A = Tables.matrix(table, transpose=true)

    lambda = learner.lambda

    # apply core algorithm:
    coefficients = (A*A' + learner.lambda*I)\(A*y) # vector

    # determine named coefficients:
    named_coefficients = [names[j] => coefficients[j] for j in eachindex(names)]

    # make some noise, if allowed:
    verbosity > 0 && @info "Coefficients: $named_coefficients"

    return RidgeFitted(learner, coefficients, named_coefficients)
end
```

## Implementing `predict`

One way users will be able to call `predict` is like this:

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
first element of the tuple returned by [`LearnAPI.kinds_of_proxy(learner)`](@ref), which
we overload appropriately below.


## Extracting the target from training data

The `fit` method consumes data which includes a [target variable](@ref proxy), i.e., the
learner is a supervised learner. We must therefore declare how the target variable can be extracted
from training data, by implementing [`LearnAPI.target`](@ref):

```@example anatomy
LearnAPI.target(learner, data) = last(data)
```

There is a similar method, [`LearnAPI.features`](@ref) for declaring how training features
can be extracted (something that can be passed to `predict`) but this method has a
fallback which suffices here: it returns `first(data)` if `data` is a tuple, and `data`
otherwise.


## Accessor functions

An [accessor function](@ref accessor_functions) has the output of [`fit`](@ref) as it's
sole argument.  Every new implementation must implement the accessor function
[`LearnAPI.learner`](@ref) for recovering a learner from a fitted object:

```@example anatomy
LearnAPI.learner(model::RidgeFitted) = model.learner
```

Other accessor functions extract learned parameters or some standard byproducts of
training, such as feature importances or training losses.² Here we implement an accessor
function to extract the linear coefficients:

```@example anatomy
LearnAPI.coefficients(model::RidgeFitted) = model.named_coefficients
nothing #hide
```

The [`LearnAPI.strip(model)`](@ref) accessor function is for returning a version of
`model` suitable for serialization (typically smaller and data anonymized). It has a
fallback that just returns `model` but for the sake of illustration, we overload it to
dump the named version of the coefficients:

```@example anatomy
LearnAPI.strip(model::RidgeFitted) =
    RidgeFitted(model.learner, model.coefficients, nothing)
```

Crucially, we can still use `LearnAPI.strip(model)` in place of `model` to make new
predictions.


## Learner traits

Learner [traits](@ref traits) record extra generic information about a learner, or
make specific promises of behavior. They are methods that have a learner as the sole
argument, and so we regard [`LearnAPI.constructor`](@ref) defined above as a trait.

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
        :(LearnAPI.learner),
        :(LearnAPI.clone),
        :(LearnAPI.strip),
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
meaningfully applied to the learner or associated model. You always include the first five
you see here: `fit`, `learner`, `clone` ,`strip`, `obs`. Here [`clone`](@ref) is a utility
function provided by LearnAPI that you never overload; overloading [`obs`](@ref) is
optional (see [Providing a separate data front end](@ref)) but it is always included
because it has a fallback. See [`LearnAPI.functions`](@ref) for a checklist.

[`LearnAPI.functions`](@ref) and [`LearnAPI.constructor`](@ref), are the only universally
compulsory traits. However, it is worthwhile studying the [list of all traits](@ref
traits_list) to see which might apply to a new implementation, to enable maximum buy into
functionality provided by third party packages, and to assist third party algorithms that
match machine learning algorithms to user-defined tasks.

Note that we know `Ridge` instances are supervised learners because `:(LearnAPI.target)
in LearnAPI.functions(learner)`, for every instance `learner`. With [some
exceptions](@ref trait_contract), the value of a trait should depend only on the *type* of
the argument.

## Signatures added for convenience

We add one `fit` signature for user-convenience only. The LearnAPI.jl specification has
nothing to say about `fit` signatures with more than two positional arguments.

```@example anatomy
LearnAPI.fit(learner::Ridge, X, y; kwargs...) = fit(learner, (X, y); kwargs...)
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
learner = Ridge(lambda=0.5)
@functions learner
```

Training and predicting:

```@example anatomy
Xtrain = Tables.subset(X, train)
ytrain = y[train]
model = fit(learner, (Xtrain, ytrain))  # `fit(learner, Xtrain, ytrain)` will also work
ŷ = predict(model, Tables.subset(X, test))
```

Extracting coefficients:

```@example anatomy
LearnAPI.coefficients(model)
```

Serialization/deserialization:

```@example anatomy
using Serialization
small_model = LearnAPI.strip(model)
filename = tempname()
serialize(filename, small_model)
```

```julia
recovered_model = deserialize(filename)
@assert LearnAPI.learner(recovered_model) == learner
@assert predict(recovered_model, X) == predict(model, X)
```

## Providing a separate data front end

```@setup anatomy2
using LearnAPI
using LinearAlgebra, Tables

struct Ridge{T<:Real}
   lambda::T
end

Ridge(; lambda=0.1) = Ridge(lambda)

struct RidgeFitted{T,F}
    learner::Ridge
    coefficients::Vector{T}
    named_coefficients::F
end

LearnAPI.learner(model::RidgeFitted) = model.learner
LearnAPI.coefficients(model::RidgeFitted) = model.named_coefficients
LearnAPI.strip(model::RidgeFitted) =
    RidgeFitted(model.learner, model.coefficients, nothing)

@trait(
    Ridge,
    constructor = Ridge,
    kinds_of_proxy=(Point(),),
    tags = (:regression,),
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.learner),
        :(LearnAPI.clone),
        :(LearnAPI.strip),
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
or `predict`, such as the matrix version `A` of `X` in the ridge example.  That is, we may
factor out of `fit` (and also `predict`) a data pre-processing step, `obs`, to expose
its outcomes. These outcomes become alternative user inputs to `fit`/`predict`.

In the default case, the alternative data representations will implement the MLUtils.jl
`getobs/numobs` interface for observation subsampling, which is generally all a user or
meta-algorithm will need, before passing the data on to `fit`/`predict` as you would the
original data.

So, instead of the pattern

```julia
model = fit(learner, data)
predict(model, newdata)
```

one enables the following alternative (which in any case will still work, because of a
no-op `obs` fallback provided by LearnAPI.jl):

```julia
observations = obs(learner, data) # pre-processed training data

# optional subsampling:
observations = MLUtils.getobs(observations, train_indices)

model = fit(learner, observations)

newobservations = obs(model, newdata)

# optional subsampling:
newobservations = MLUtils.getobs(observations, test_indices)

predict(model, newobservations)
```

See also the demonstration [below](@ref advanced_demo).

Here we specifically wrap all the pre-processed data into single object, for which we
introduce a new type:

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
function LearnAPI.fit(learner::Ridge, observations::RidgeFitObs; verbosity=LearnAPI.default_verbosity())

    lambda = learner.lambda

    A = observations.A
    names = observations.names
    y = observations.y

    # apply core learner:
    coefficients = (A*A' + learner.lambda*I)\(A*y) # 1 x p matrix

    # determine named coefficients:
    named_coefficients = [names[j] => coefficients[j] for j in eachindex(names)]

    # make some noise, if allowed:
    verbosity > 0 && @info "Coefficients: $named_coefficients"

    return RidgeFitted(learner, coefficients, named_coefficients)

end

LearnAPI.fit(learner::Ridge, data; kwargs...) =
    fit(learner, obs(learner, data); kwargs...)
```

### The `obs` contract

Providing `fit` signatures matching the output of [`obs`](@ref), is the first part of the
`obs` contract. Since `obs(learner, data)` should evidently support all `data` that
`fit(learner, data)` supports, we must be able to apply `obs(learner, _)` to it's own
output (`observations` below). This leads to the additional "no-op" declaration

```@example anatomy2
LearnAPI.obs(::Ridge, observations::RidgeFitObs) = observations
```

In other words, we ensure that `obs(learner, _)` is
[involutive](https://en.wikipedia.org/wiki/Involution_(mathematics)).

The second part of the `obs` contract is this: *The output of `obs` must implement the
interface specified by the trait* [`LearnAPI.data_interface(learner)`](@ref). Assuming
this is [`LearnAPI.RandomAccess()`](@ref) (the default) it usually suffices to overload
`Base.getindex` and `Base.length`:

```@example anatomy2
Base.getindex(data::RidgeFitObs, I) =
    RidgeFitObs(data.A[:,I], data.names, y[I])
Base.length(data::RidgeFitObs) = length(data.y)
```

We do something similar for `predict`, but there's no need for a new type in this case:

```@example anatomy2
LearnAPI.obs(::RidgeFitted, Xnew) = Tables.matrix(Xnew)'
LearnAPI.obs(::RidgeFitted, observations::AbstractArray) = observations # involutivity

LearnAPI.predict(model::RidgeFitted, ::Point, observations::AbstractMatrix) =
    observations'*model.coefficients

LearnAPI.predict(model::RidgeFitted, ::Point, Xnew) =
    predict(model, Point(), obs(model, Xnew))
```

### `target` and `features` methods

In the general case, we only need to implement [`LearnAPI.target`](@ref) and
[`LearnAPI.features`](@ref) to handle all possible output of `obs(learner, data)`, and now
the fallback for `LearnAPI.features` mentioned before is inadequate.

```@example anatomy2
LearnAPI.target(::Ridge, observations::RidgeFitObs) = observations.y
LearnAPI.features(::Ridge, observations::RidgeFitObs) = observations.A
```

### Important notes:

- The observations to be consumed by `fit` are returned by `obs(learner::Ridge, ...)`,
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

To opt out of supporting the MLUtils.jl interface altogether, an implementation must
overload the trait, [`LearnAPI.data_interface(learner)`](@ref). See [Data
interfaces](@ref data_interfaces) for details.


### Addition of signatures for user convenience

As above, we add a signature for convenience, which the LearnAPI.jl specification
neither requires nor forbids:

```@example anatomy2
LearnAPI.fit(learner::Ridge, X, y; kwargs...)  = fit(learner, (X, y); kwargs...)
```

## [Demonstration of an advanced `obs` workflow](@id advanced_demo)

We now can train and predict using internal data representations, resampled using the
generic MLUtils.jl interface:

```@example anatomy2
import MLUtils
learner = Ridge()
observations_for_fit = obs(learner, (X, y))
model = fit(learner, MLUtils.getobs(observations_for_fit, train))
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

⁴ The `data = (X, y)` pattern implemented here is not the only supported pattern. For,
example, `data` might be `(T, formula)` where `T` is a table and `formula` is an R-style
formula.
