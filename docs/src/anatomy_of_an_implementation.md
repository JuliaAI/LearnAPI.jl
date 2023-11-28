# Anatomy of an Implementation

This section explains a detailed implementation of the LearnAPI for naive [ridge
regression](https://en.wikipedia.org/wiki/Ridge_regression).  Most readers will want to
scan the [demonstration](@ref workflow) of the implementation before studying the
implementation itself.

## Defining an algorithm type

The first line below imports the lightweight package LearnAPI.jl whose methods we will be
extending. The second imports libraries needed for the core algorithm.

```@example anatomy
using LearnAPI
using LinearAlgebra, Tables
nothing # hide
```

A struct stores the regularization hyperparameter `lambda` of our ridge regressor:

```@example anatomy
struct Ridge
    lambda::Float64
end
nothing # hide
```

Instances of `Ridge` are [algorithms](@ref algorithms), in LearnAPI.jl parlance.

A keyword argument constructor provides defaults for all hyperparameters:

```@example anatomy
Ridge(; lambda=0.1) = Ridge(lambda)
nothing # hide
```

## Implementing `fit`

A ridge regressor requires two types of data for training: *input features* `X`, which
here we suppose are tabular, and a [target](@ref proxy) `y`, which we suppose is a
vector. Users will accordingly call [`fit`](@ref) like this:

```julia
algorithm = Ridge(lambda=0.05)
fit(algorithm, X, y; verbosity=1)
```

However, a new implementation does not overload `fit`. Rather it
implements

```julia
obsfit(algorithm::Ridge, obsdata; verbosity=1)
```

for each `obsdata` returned by a data-preprocessing call `obs(fit, algorithm, X, y)`. You
can read "obs" as "observation-accessible", for reasons explained shortly. The
LearnAPI.jl definition

```julia
fit(algorithm, data...; verbosity=1) =
    obsfit(algorithm, obs(fit, algorithm, data...), verbosity)
```
then takes care of `fit`.

The `obs` and `obsfit`  method are public, and the user can call them like this:

```julia
obsdata = obs(fit, algorithm, X, y)
model = obsfit(algorithm, obsdata)
```

We begin by defining a struct¹ for the output of our data-preprocessing operation, `obs`,
which will store `y` and the matrix representation of `X`, together with it's column names
(needed for recording named coefficients for user inspection):

```@example anatomy
struct RidgeFitData{T}
    A::Matrix{T}    # p x n
    names::Vector{Symbol}
    y::Vector{T}
end
nothing # hide
```

And we overload [`obs`](@ref) like this

```@example anatomy
function LearnAPI.obs(::typeof(fit), ::Ridge, X, y)
    table = Tables.columntable(X)
    names = Tables.columnnames(table) |> collect
    return RidgeFitData(Tables.matrix(table, transpose=true), names, y)
end
nothing # hide
```

so that `obs(fit, Ridge(), X, y)` returns a combined `RidgeFitData` object with everything
the core algorithm will need.

Since `obs` is public, the user will have access to this object, but to make it useful to
her (and to fulfill the [`obs`](@ref) contract) this object must implement the
[MLUtils.jl](https://github.com/JuliaML/MLUtils.jl) `getobs`/`numobs` interface, to enable
observation-resampling (which will be efficient, because observations are now columns). It
usually suffices to overload `Base.getindex` and `Base.length` (which are the
`getobs`/`numobs` fallbacks) so we won't actually need to depend on MLUtils.jl:

```@example anatomy
Base.getindex(data::RidgeFitData, I) =
    RidgeFitData(data.A[:,I], data.names, y[I])
Base.length(data::RidgeFitData, I) = length(data.y)
nothing # hide
```

Next, we define a second struct for storing the outcomes of training, including named
versions of the learned coefficients:

```@example anatomy
struct RidgeFitted{T,F}
    algorithm::Ridge
    coefficients::Vector{T}
    named_coefficients::F
end
nothing # hide
```

We include `algorithm`, which must be recoverable from the output of `fit`/`obsfit` (see
[Accessor functions](@ref) below).

We are now ready to implement a suitable `obsfit` method to execute the core training:

```@example anatomy
function LearnAPI.obsfit(algorithm::Ridge, obsdata::RidgeFitData, verbosity)

    lambda = algorithm.lambda
    A = obsdata.A
    names = obsdata.names
    y = obsdata.y

    # apply core algorithm:
    coefficients = (A*A' + algorithm.lambda*I)\(A*y) # 1 x p matrix

    # determine named coefficients:
    named_coefficients = [names[j] => coefficients[j] for j in eachindex(names)]

    # make some noise, if allowed:
    verbosity > 0 && @info "Coefficients: $named_coefficients"

    return RidgeFitted(algorithm, coefficients, named_coefficients)

end
nothing # hide
```
Users set `verbosity=0` for warnings only, and `verbosity=-1` for silence.


## Implementing `predict`

The primary `predict` call will look like this:

```julia
predict(model, LiteralTarget(), Xnew)
```

where `Xnew` is a table (of the same form as `X` above). The argument `LiteralTarget()`
signals that we want literal predictions of the target variable, as opposed to a proxy for
the target, such as probability density functions.  `LiteralTarget` is an example of a
[`LearnAPI.KindOfProxy`](@ref proxy_types) type. Targets and target proxies are defined
[here](@ref proxy).

Rather than overload the primary signature above, however, we overload for
"observation-accessible" input, as we did for `fit`,

```@example anatomy
LearnAPI.obspredict(model::RidgeFitted, ::LiteralTarget, Anew::Matrix) =
    ((model.coefficients)'*Anew)'
nothing # hide
```

and overload `obs` to make the table-to-matrix conversion:

```@example anatomy
LearnAPI.obs(::typeof(predict), ::Ridge, Xnew) = Tables.matrix(Xnew, transpose=true)
```

As matrices (with observations as columns) already implement the MLUtils.jl
`getobs`/`numobs` interface, we already satisfy the [`obs`](@ref) contract, and there was
no need to create a wrapper for `obs` output.

The primary `predict` method, handling tabular input, is provided by a
LearnAPI.jl fallback similar to the `fit` fallback.


## Accessor functions

An [accessor function](@ref accessor_functions) has the output of [`fit`](@ref) (a
"model") as it's sole argument.  Every new implementation must implement the accessor
function [`LearnAPI.algorithm`](@ref) for recovering an algorithm from a fitted object:

```@example anatomy
LearnAPI.algorithm(model::RidgeFitted) = model.algorithm
```

Other accessor functions extract learned parameters or some standard byproducts of
training, such as feature importances or training losses.² Implementing the
[`LearnAPI.coefficients`](@ref) accessor function is straightforward:

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

## Algorithm traits

Algorithm [traits](@ref traits) record extra generic information about an algorithm, or
make specific promises of behavior. They usually have an algorithm as the single argument.

In LearnAPI.jl `predict` always outputs a [target or target proxy](@ref proxy), where
"target" is understood very broadly. We overload a trait to record the fact that the
target variable explicitly appears in training (i.e, the algorithm is supervised) and
where exactly it appears:

```julia
LearnAPI.position_of_target(::Ridge) = 2
```
Or, you can use the shorthand

```julia
@trait Ridge position_of_target = 2
```

The macro can also be used to specify multiple traits simultaneously:

```@example anatomy
@trait(
    Ridge,
    position_of_target = 2,
    kinds_of_proxy=(LiteralTarget(),),
    descriptors = (:regression,),
    functions = (
        fit,
        obsfit,
        minimize,
        predict,
        obspredict,
        obs,
        LearnAPI.algorithm,
        LearnAPI.coefficients,
    )
)
nothing # hide
```

Implementing the last trait, [`LearnAPI.functions`](@ref), which must include all
non-trait functions overloaded for `Ridge`, is compulsory. This is the only universally
compulsory trait. It is worthwhile studying the [list of all traits](@ref traits_list) to
see which might apply to a new implementation, to enable maximum buy into functionality
provided by third party packages, and to assist third party algorithms that match machine
learning algorithms to user defined tasks.

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

algorithm = Ridge(lambda=0.5)
LearnAPI.functions(algorithm)
```

### Naive user workflow

Training and predicting with external resampling:

```@example anatomy
using Tables
model = fit(algorithm, Tables.subset(X, train), y[train])
ŷ = predict(model, LiteralTarget(), Tables.subset(X, test))
```

### Advanced workflow

We now train and predict using internal data representations, resampled using the generic
MLUtils.jl interface.

```@example anatomy
import MLUtils
fit_data = obs(fit, algorithm, X, y)
predict_data = obs(predict, algorithm, X)
model = obsfit(algorithm, MLUtils.getobs(fit_data, train))
ẑ = obspredict(model, LiteralTarget(), MLUtils.getobs(predict_data, test))
@assert ẑ == ŷ
nothing # hide
```

### Applying an accessor function and serialization

Extracting coefficients:

```@example anatomy
LearnAPI.coefficients(model)
```

Serialization/deserialization:

```julia
using Serialization
small_model = minimize(model)
serialize("my_ridge.jls", small_model)

recovered_model = deserialize("my_ridge.jls")
@assert LearnAPI.algorithm(recovered_model) == algorithm
predict(recovered_model, LiteralTarget(), X) == predict(model, LiteralTarget(), X)
```

---

¹ The definition of this and other structs above is not an explicit requirement of
LearnAPI.jl, whose constructs are purely functional. 

² An implementation can provide further accessor functions, if necessary, but
like the native ones, they must be included in the [`LearnAPI.functions`](@ref)
declaration.
