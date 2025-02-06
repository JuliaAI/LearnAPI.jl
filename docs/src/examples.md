# [Code for ridge example](@id code)

Below is the complete source code for the ridge implementations described in the tutorial,
[Anatomy of an Implementation](@ref).

- [Basic implementation](@ref)
- [Implementation with data front end](@ref)


## Basic implementation

```julia
using LearnAPI
using LinearAlgebra, Tables

struct Ridge{T<:Real}
    lambda::T
end

"""
    Ridge(; lambda=0.1)

Instantiate a ridge regression learner, with regularization of `lambda`.
"""
Ridge(; lambda=0.1) = Ridge(lambda)
LearnAPI.constructor(::Ridge) = Ridge

# struct for output of `fit`
struct RidgeFitted{T,F}
    learner::Ridge
    coefficients::Vector{T}
    named_coefficients::F
end

function LearnAPI.fit(learner::Ridge, data; verbosity=1)
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

LearnAPI.predict(model::RidgeFitted, ::Point, Xnew) =
    Tables.matrix(Xnew)*model.coefficients

# accessor functions:
LearnAPI.learner(model::RidgeFitted) = model.learner
LearnAPI.coefficients(model::RidgeFitted) = model.named_coefficients
LearnAPI.strip(model::RidgeFitted) =
    RidgeFitted(model.learner, model.coefficients, nothing)

@trait(
    Ridge,
    constructor = Ridge,
    kinds_of_proxy=(Point(),),
    tags = ("regression",),
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

# convenience method:
LearnAPI.fit(learner::Ridge, X, y; kwargs...) = fit(learner, (X, y); kwargs...)
```

# Implementation with data front end

```julia
using LearnAPI
using LinearAlgebra, Tables

struct Ridge{T<:Real}
   lambda::T
end

Ridge(; lambda=0.1) = Ridge(lambda)

# struct for output of `fit`:
struct RidgeFitted{T,F}
    learner::Ridge
    coefficients::Vector{T}
    named_coefficients::F
end

# struct for internal representation of training data:
struct RidgeFitObs{T,M<:AbstractMatrix{T}}
    A::M                  # `p` x `n` matrix
    names::Vector{Symbol} # features
    y::Vector{T}          # target
end

# implementation of `RandomAccess()` data interface for such representation:
Base.getindex(data::RidgeFitObs, I) =
    RidgeFitObs(data.A[:,I], data.names, y[I])
Base.length(data::RidgeFitObs) = length(data.y)

# data front end for `fit`:
function LearnAPI.obs(::Ridge, data)
    X, y = data
    table = Tables.columntable(X)
    names = Tables.columnnames(table) |> collect
    return RidgeFitObs(Tables.matrix(table)', names, y)
end
LearnAPI.obs(::Ridge, observations::RidgeFitObs) = observations

function LearnAPI.fit(learner::Ridge, observations::RidgeFitObs; verbosity=1)

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

# data front end for `predict`:
LearnAPI.obs(::RidgeFitted, Xnew) = Tables.matrix(Xnew)'
LearnAPI.obs(::RidgeFitted, observations::AbstractArray) = observations # involutivity

LearnAPI.predict(model::RidgeFitted, ::Point, observations::AbstractMatrix) =
    observations'*model.coefficients

LearnAPI.predict(model::RidgeFitted, ::Point, Xnew) =
    predict(model, Point(), obs(model, Xnew))

# methods to deconstruct training data:
LearnAPI.features(::Ridge, observations::RidgeFitObs) = observations.A
LearnAPI.target(::Ridge, observations::RidgeFitObs) = observations.y
LearnAPI.features(learner::Ridge, data) = LearnAPI.features(learner, obs(learner, data))
LearnAPI.target(learner::Ridge, data) = LearnAPI.target(learner, obs(learner, data))

# accessor functions:
LearnAPI.learner(model::RidgeFitted) = model.learner
LearnAPI.coefficients(model::RidgeFitted) = model.named_coefficients
LearnAPI.strip(model::RidgeFitted) =
    RidgeFitted(model.learner, model.coefficients, nothing)

@trait(
    Ridge,
    constructor = Ridge,
    kinds_of_proxy=(Point(),),
    tags = ("regression",),
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

```
