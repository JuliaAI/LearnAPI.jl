using LearnAPI
using LinearAlgebra
using Tables
import MLUtils
import DataFrames


# # NAIVE RIDGE REGRESSION WITH NO INTERCEPTS

# We overload `obs` to expose internal representation of data. See later for a simpler
# variation using the `obs` fallback.

# no docstring here - that goes with the constructor
struct Ridge
    lambda::Float64
end

"""
    Ridge(; lambda=0.1)

Instantiate a ridge regression algorithm, with regularization of `lambda`.

"""
Ridge(; lambda=0.1) = Ridge(lambda) # LearnAPI.constructor defined later

struct RidgeFitObs{T,M<:AbstractMatrix{T}}
    A::M  # p x n
    names::Vector{Symbol}
    y::Vector{T}
end

struct RidgeFitted{T,F}
    algorithm::Ridge
    coefficients::Vector{T}
    feature_importances::F
end

LearnAPI.algorithm(model::RidgeFitted) = model.algorithm

Base.getindex(data::RidgeFitObs, I) =
    RidgeFitObs(data.A[:,I], data.names, data.y[I])
Base.length(data::RidgeFitObs, I) = length(data.y)

# observations for consumption by `fit`:
function LearnAPI.obs(::Ridge, data)
    X, y = data
    table = Tables.columntable(X)
    names = Tables.columnnames(table) |> collect
    RidgeFitObs(Tables.matrix(table)', names, y)
end

# for observations:
function LearnAPI.fit(algorithm::Ridge, observations::RidgeFitObs; verbosity=1)

    # unpack hyperparameters and data:
    lambda = algorithm.lambda
    A = observations.A
    names = observations.names
    y = observations.y

    # apply core algorithm:
    coefficients = (A*A' + algorithm.lambda*I)\(A*y) # 1 x p matrix

    # determine crude feature importances:
    feature_importances =
        [names[j] => abs(coefficients[j]) for j in eachindex(names)]
    sort!(feature_importances, by=last) |> reverse!

    # make some noise, if allowed:
    verbosity > 0 &&
        @info "Features in order of importance: $(first.(feature_importances))"

    return RidgeFitted(algorithm, coefficients, feature_importances)

end

# for unprocessed `data = (X, y)`:
LearnAPI.fit(algorithm::Ridge, data; kwargs...) =
    fit(algorithm, obs(algorithm, data); kwargs...)

# extracting stuff from training data:
LearnAPI.target(::Ridge, data) = last(data)
LearnAPI.target(::Ridge, observations::RidgeFitObs) = observations.y
LearnAPI.input(::Ridge, observations::RidgeFitObs) = observations.A

# observations for consumption by `predict`:
LearnAPI.obs(::RidgeFitted, X) = Tables.matrix(X)'

# matrix input:
LearnAPI.predict(model::RidgeFitted, ::LiteralTarget, observations::AbstractMatrix) =
        observations'*model.coefficients

# tabular input:
LearnAPI.predict(model::RidgeFitted, ::LiteralTarget, Xnew) =
        predict(model, LiteralTarget(), obs(model, Xnew))

# accessor function:
LearnAPI.feature_importances(model::RidgeFitted) = model.feature_importances

LearnAPI.minimize(model::RidgeFitted) =
    RidgeFitted(model.algorithm, model.coefficients, nothing)

@trait(
    Ridge,
    constructor = Ridge,
    kinds_of_proxy = (LiteralTarget(),),
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.algorithm),
        :(LearnAPI.minimize),
        :(LearnAPI.obs),
        :(LearnAPI.input),
        :(LearnAPI.target),
        :(LearnAPI.predict),
        :(LearnAPI.feature_importances),
   )
)

# synthetic test data:
n = 30 # number of observations
train = 1:6
test = 7:10
a, b, c = rand(n), rand(n), rand(n)
X = (; a, b, c)
X = DataFrames.DataFrame(X)
y = 2a - b + 3c + 0.05*rand(n)
data = (X, y)

@testset "test an implementation of ridge regression" begin
    algorithm = Ridge(lambda=0.5)
    @test :(LearnAPI.obs) in LearnAPI.functions(algorithm)

    @test LearnAPI.target(algorithm, data) == y
    @test LearnAPI.input(algorithm, data) == X

    # verbose fitting:
    @test_logs(
        (:info, r"Feature"),
        fit(
            algorithm,
            Tables.subset(X, train),
            y[train];
            verbosity=1,
        ),
    )

    # quiet fitting:
    model = @test_logs(
        fit(
            algorithm,
            Tables.subset(X, train),
            y[train];
            verbosity=0,
        ),
    )

    ŷ = predict(model, LiteralTarget(), Tables.subset(X, test))
    @test ŷ isa Vector{Float64}
    @test predict(model, Tables.subset(X, test)) == ŷ

    fitobs = LearnAPI.obs(algorithm, data)
    predictobs = LearnAPI.obs(model, X)
    model = fit(algorithm, MLUtils.getobs(fitobs, train); verbosity=0)
    @test LearnAPI.target(algorithm, fitobs) == y
    @test predict(model, LiteralTarget(), MLUtils.getobs(predictobs, test)) ≈ ŷ
    @test predict(model, LearnAPI.input(algorithm, fitobs)) ≈ predict(model, X)

    @test LearnAPI.feature_importances(model) isa Vector{<:Pair{Symbol}}

    filename = tempname()
    using Serialization
    small_model = minimize(model)
    serialize(filename, small_model)

    recovered_model = deserialize(filename)
    @test LearnAPI.algorithm(recovered_model) == algorithm
    @test predict(
        recovered_model,
        LiteralTarget(),
        MLUtils.getobs(predictobs, test)
    ) ≈ ŷ

end

# # VARIATION OF RIDGE REGRESSION THAT USES FALLBACK OF LearnAPI.obs

# no docstring here - that goes with the constructor
struct BabyRidge
    lambda::Float64
end

"""
    BabyRidge(; lambda=0.1)

Instantiate a ridge regression algorithm, with regularization of `lambda`.

"""
BabyRidge(; lambda=0.1) = BabyRidge(lambda) # LearnAPI.constructor defined later

struct BabyRidgeFitted{T,F}
    algorithm::BabyRidge
    coefficients::Vector{T}
    feature_importances::F
end

function LearnAPI.fit(algorithm::BabyRidge, data; verbosity=1)

    X, y = data

    lambda = algorithm.lambda
    table = Tables.columntable(X)
    names = Tables.columnnames(table) |> collect
    A = Tables.matrix(table)'

    # apply core algorithm:
    coefficients = (A*A' + algorithm.lambda*I)\(A*y) # vector

    feature_importances = nothing

    return BabyRidgeFitted(algorithm, coefficients, feature_importances)

end

# extracting stuff from training data:
LearnAPI.target(::BabyRidge, data) = last(data)

LearnAPI.algorithm(model::BabyRidgeFitted) = model.algorithm

LearnAPI.predict(model::BabyRidgeFitted, ::LiteralTarget, Xnew) =
    Tables.matrix(Xnew)*model.coefficients

LearnAPI.minimize(model::BabyRidgeFitted) =
    BabyRidgeFitted(model.algorithm, model.coefficients, nothing)

@trait(
    BabyRidge,
    constructor = BabyRidge,
    kinds_of_proxy = (LiteralTarget(),),
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.algorithm),
        :(LearnAPI.minimize),
        :(LearnAPI.obs),
        :(LearnAPI.input),
        :(LearnAPI.target),
        :(LearnAPI.predict),
        :(LearnAPI.feature_importances),
   )
)

@testset "test a variation  which does not overload LearnAPI.obs" begin
    algorithm = BabyRidge(lambda=0.5)

    model = fit(algorithm, Tables.subset(X, train), y[train]; verbosity=0)
    ŷ = predict(model, LiteralTarget(), Tables.subset(X, test))
    @test ŷ isa Vector{Float64}

    fitobs = obs(algorithm, data)
    predictobs = LearnAPI.obs(model, X)
    model = fit(algorithm, MLUtils.getobs(fitobs, train); verbosity=0)
    @test predict(model, LiteralTarget(), MLUtils.getobs(predictobs, test)) == ŷ ==
        predict(model, MLUtils.getobs(predictobs, test))
    @test LearnAPI.target(algorithm, data) == y
    @test LearnAPI.predict(model, X) ≈
        LearnAPI.predict(model, LearnAPI.input(algorithm, data))
end

true
