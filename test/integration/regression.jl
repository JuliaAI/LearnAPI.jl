using LearnAPI
using LinearAlgebra
using Tables
import MLUtils
import DataFrames


# # NAIVE RIDGE REGRESSION WITH NO INTERCEPTS

# We overload `obs` to expose internal representation of input data. See later for a
# simpler variation using the `obs` fallback.

struct Ridge
    lambda::Float64
end
Ridge(; lambda=0.1) = Ridge(lambda)

struct RidgeFitObs{T}
    A::Matrix{T}    # p x n
    names::Vector{Symbol}
    y::Vector{T}
end

struct RidgeFitted{T,F}
    algorithm::Ridge
    coefficients::Vector{T}
    feature_importances::F
end

Base.getindex(data::RidgeFitObs, I) =
    RidgeFitObs(data.A[:,I], data.names, data.y[I])
Base.length(data::RidgeFitObs, I) = length(data.y)

function LearnAPI.obs(::typeof(fit), ::Ridge, X, y)
    table = Tables.columntable(X)
    names = Tables.columnnames(table) |> collect
    RidgeFitObs(Tables.matrix(table, transpose=true), names, y)
end

function LearnAPI.obsfit(algorithm::Ridge, fitdata::RidgeFitObs, verbosity)

    # unpack hyperparameters and data:
    lambda = algorithm.lambda
    A = fitdata.A
    names = fitdata.names
    y = fitdata.y

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

LearnAPI.algorithm(model::RidgeFitted) = model.algorithm

LearnAPI.obspredict(model::RidgeFitted, ::LiteralTarget, Anew::Matrix) =
    ((model.coefficients)'*Anew)'

LearnAPI.obs(::typeof(predict), ::Ridge, X) = Tables.matrix(X, transpose=true)

LearnAPI.feature_importances(model::RidgeFitted) = model.feature_importances

LearnAPI.minimize(model::RidgeFitted) =
    RidgeFitted(model.algorithm, model.coefficients, nothing)

@trait(
    Ridge,
    position_of_target=2,
    kinds_of_proxy = (LiteralTarget(),),
    functions = (
        fit,
        obsfit,
        minimize,
        predict,
        obspredict,
        obs,
        LearnAPI.algorithm,
        LearnAPI.feature_importances,
    )
)

n = 10 # number of observations
train = 1:6
test = 7:10
a, b, c = rand(n), rand(n), rand(n)
X = (; a, b, c)
X = DataFrames.DataFrame(X)
y = 2a - b + 3c + 0.05*rand(n)

@testset "test an implementation of ridge regression" begin
    algorithm = Ridge(lambda=0.5)
    @test LearnAPI.obs in LearnAPI.functions(algorithm)

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

    # quite fitting:
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

    fitdata = LearnAPI.obs(fit, algorithm, X, y)
    predictdata = LearnAPI.obs(predict, algorithm, X)
    model = obsfit(algorithm, MLUtils.getobs(fitdata, train); verbosity=1)
    @test obspredict(model, LiteralTarget(), MLUtils.getobs(predictdata, test)) == ŷ

    @test LearnAPI.feature_importances(model) isa Vector{<:Pair{Symbol}}

    filename = tempname()
    using Serialization
    small_model = minimize(model)
    serialize(filename, small_model)

    recovered_model = deserialize(filename)
    @test LearnAPI.algorithm(recovered_model) == algorithm
    @test obspredict(
        recovered_model,
        LiteralTarget(),
        MLUtils.getobs(predictdata, test)
    ) == ŷ
end

# # VARIATION OF RIDGE REGRESSION THAT USES FALLBACK OF LearnAPI.obs

struct BabyRidge
    lambda::Float64
end
BabyRidge(; lambda=0.1) = BabyRidge(lambda)

struct BabyRidgeFitted{T,F}
    algorithm::BabyRidge
    coefficients::Vector{T}
    feature_importances::F
end

function LearnAPI.obsfit(algorithm::BabyRidge, data, verbosity)

    X, y = data

    lambda = algorithm.lambda

    table = Tables.columntable(X)
    names = Tables.columnnames(table) |> collect
    A = Tables.matrix(table, transpose=true)

    # apply core algorithm:
    coefficients = (A*A' + algorithm.lambda*I)\(A*y) # 1 x p matrix

    feature_importances = nothing

    return BabyRidgeFitted(algorithm, coefficients, feature_importances)

end

LearnAPI.algorithm(model::BabyRidgeFitted) = model.algorithm

function LearnAPI.obspredict(model::BabyRidgeFitted, ::LiteralTarget, data)
    X = only(data)
    Anew = Tables.matrix(X, transpose=true)
    return ((model.coefficients)'*Anew)'
end

@trait(
    BabyRidge,
    position_of_target=2,
    kinds_of_proxy = (LiteralTarget(),),
    functions = (
        fit,
        obsfit,
        minimize,
        predict,
        obspredict,
        obs,
        LearnAPI.algorithm,
        LearnAPI.feature_importances,
    )
)

@testset "test a variation  which does not overload LearnAPI.obs" begin
    algorithm = BabyRidge(lambda=0.5)

    model = fit(algorithm, Tables.subset(X, train), y[train]; verbosity=0)
    ŷ = predict(model, LiteralTarget(), Tables.subset(X, test))
    @test ŷ isa Vector{Float64}

    fitdata = obs(fit, algorithm, X, y)
    predictdata = LearnAPI.obs(predict, algorithm, X)
    model = obsfit(algorithm, MLUtils.getobs(fitdata, train); verbosity=0)
    @test obspredict(model, LiteralTarget(), MLUtils.getobs(predictdata, test)) == ŷ
end

true
