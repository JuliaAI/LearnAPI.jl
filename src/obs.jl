"""
    obs(func, algorithm, data...)

Where `func` is `fit`, `predict` or `transform`, return a combined, algorithm-specific,
representation of `data...`, which can be passed directly to `obsfit`, `obspredict` or
`obstransform`, as shown in the example below.

The returned object implements the `getobs`/`numobs` observation-resampling interface
provided by MLUtils.jl, even if `data` does not.

Calling `func` on the returned object may be cheaper than calling `func` directly on
`data...`. And resampling the returned object using `MLUtils.getobs` may be cheaper than
directly resampling the components of `data` (an operation not provided by the LearnAPI.jl
interface).

# Example

Usual workflow, using data-specific resampling methods:

```julia
X = <some `DataFrame`>
y = <some `Vector`>

Xtrain = Tables.select(X, 1:100)
ytrain = y[1:100]
model = fit(algorithm, Xtrain, ytrain)
ŷ = predict(model, LiteralTarget(), y[101:150])
```

Alternative workflow using `obs`:

```julia
import MLUtils

fitdata = obs(fit, algorithm, X, y)
predictdata = obs(predict, algorithm, X)

model = obsfit(algorithm, MLUtils.getobs(fitdata, 1:100))
ẑ = obspredict(model, LiteralTarget(), MLUtils.getobs(predictdata, 101:150))
@assert ẑ == ŷ
```

See also [`obsfit`](@ref), [`obspredict`](@ref), [`obstransform`](@ref).


# Extended help

# New implementations

If the `data` to be consumed in standard user calls to `fit`, `predict` or `transform`
consists only of tables and arrays (with last dimension the observation dimension) then
overloading `obs` is optional, but the user will get no performance benefits by using
it. The implementation of `obs` is optional under more general circumstances stated at the
end.

The fallback for `obs` just slurps the provided data:

```julia
obs(func, alg, data...) = data
```

The only contractual obligation of `obs` is to return an object implementing the
`getobs`/`numobs` interface. Generally it suffices to overload `Base.getindex` and
`Base.length`. However, note that implementations of [`obsfit`](@ref),
[`obspredict`](@ref), and [`obstransform`](@ref) depend on the form of output of `obs`.

$(DOC_IMPLEMENTED_METHODS(:(obs), overloaded=true))

## Sample implementation

Suppose that `fit`, for an algorithm of type `Alg`, is to have the primary signature

```julia
fit(algorithm::Alg, X, y)
```

where `X` is a table, `y` a vector. Internally, the algorithm is to call a lower level
function

`train(A, names, y)`

where `A = Tables.matrix(X)'` and `names` are the column names of `X`. Then relevant parts
of an implementation might look like this:

```julia
# thin wrapper for algorithm-specific representation of data:
struct ObsData{T}
    A::Matrix{T}
    names::Vector{Symbol}
    y::Vector{T}
end

# (indirect) implementation of `getobs/numobs`:
Base.getindex(data::ObsData, I) =
    ObsData(data.A[:,I], data.names, y[I])
Base.length(data::ObsData, I) = length(data.y)

# implementation of `obs`:
function LearnAPI.obs(::typeof(fit), ::Alg, X, y)
    table = Tables.columntable(X)
    names = Tables.columnnames(table) |> collect
    return ObsData(Tables.matrix(table)', names, y)
end

# implementation of `obsfit`:
function LearnAPI.obsfit(algorithm::Alg, data::ObsData; verbosity=1)
    coremodel = train(data.A, data.names, data.y)
    data.verbosity > 0 && @info "Training using these features: $names."
    <construct final `model` using `coremodel`>
    return model
end
```

## When is overloading `obs` optional?

Overloading `obs` is optional, for a given `typeof(algorithm)` and `typeof(fun)`, if the
components of `data` in the standard call `func(algorithm_or_model, data...)` are already
expected to separately implement the `getobs`/`numbobs` interface. This is true for arrays
whose last dimension is the observation dimension, and for suitable tables.

"""
obs(func, alg, data...) = data
