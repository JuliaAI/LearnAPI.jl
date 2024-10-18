using LearnAPI
using LinearAlgebra
using Tables
import MLUtils
import DataFrames


# # NAIVE RIDGE REGRESSION WITH NO INTERCEPTS

# We overload `obs` to expose internal representation of data. See later for a simpler
# variation using the `obs` fallback.


# ## Implementation

# no docstring here - that goes with the constructor
struct Ridge
    lambda::Float64
end

"""
    Ridge(; lambda=0.1)

Instantiate a ridge regression learner, with regularization of `lambda`.

"""
Ridge(; lambda=0.1) = Ridge(lambda) # LearnAPI.constructor defined later

struct RidgeFitObs{T,M<:AbstractMatrix{T}}
    A::M  # p x n
    names::Vector{Symbol}
    y::Vector{T}
end

struct RidgeFitted{T,F}
    learner::Ridge
    coefficients::Vector{T}
    feature_importances::F
end

LearnAPI.learner(model::RidgeFitted) = model.learner

Base.getindex(data::RidgeFitObs, I) =
    RidgeFitObs(data.A[:,I], data.names, data.y[I])
Base.length(data::RidgeFitObs) = length(data.y)

# observations for consumption by `fit`:
function LearnAPI.obs(::Ridge, data)
    X, y = data
    table = Tables.columntable(X)
    names = Tables.columnnames(table) |> collect
    RidgeFitObs(Tables.matrix(table)', names, y)
end

# for observations:
function LearnAPI.fit(learner::Ridge, observations::RidgeFitObs; verbosity=1)

    # unpack hyperparameters and data:
    lambda = learner.lambda
    A = observations.A
    names = observations.names
    y = observations.y

    # apply core learner:
    coefficients = (A*A' + learner.lambda*I)\(A*y) # 1 x p matrix

    # determine crude feature importances:
    feature_importances =
        [names[j] => abs(coefficients[j]) for j in eachindex(names)]
    sort!(feature_importances, by=last) |> reverse!

    # make some noise, if allowed:
    verbosity > 0 &&
        @info "Features in order of importance: $(first.(feature_importances))"

    return RidgeFitted(learner, coefficients, feature_importances)

end

# for unprocessed `data = (X, y)`:
LearnAPI.fit(learner::Ridge, data; kwargs...) =
    fit(learner, obs(learner, data); kwargs...)

# extracting stuff from training data:
LearnAPI.target(::Ridge, data) = last(data)
LearnAPI.target(::Ridge, observations::RidgeFitObs) = observations.y
LearnAPI.features(::Ridge, observations::RidgeFitObs) = observations.A

# observations for consumption by `predict`:
LearnAPI.obs(::RidgeFitted, X) = Tables.matrix(X)'

# matrix input:
LearnAPI.predict(model::RidgeFitted, ::Point, observations::AbstractMatrix) =
        observations'*model.coefficients

# tabular input:
LearnAPI.predict(model::RidgeFitted, ::Point, Xnew) =
        predict(model, Point(), obs(model, Xnew))

# accessor function:
LearnAPI.feature_importances(model::RidgeFitted) = model.feature_importances

LearnAPI.strip(model::RidgeFitted) =
    RidgeFitted(model.learner, model.coefficients, nothing)

@trait(
    Ridge,
    constructor = Ridge,
    kinds_of_proxy = (Point(),),
    tags = ("regression",),
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.learner),
        :(LearnAPI.strip),
        :(LearnAPI.obs),
        :(LearnAPI.features),
        :(LearnAPI.target),
        :(LearnAPI.predict),
        :(LearnAPI.feature_importances),
   )
)

# convenience method:
LearnAPI.fit(learner::Ridge, X, y; kwargs...) =
    fit(learner, (X, y); kwargs...)


# ## Tests

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
    learner = Ridge(lambda=0.5)
    @test :(LearnAPI.obs) in LearnAPI.functions(learner)

    @test LearnAPI.target(learner, data) == y
    @test LearnAPI.features(learner, data) == X

    # verbose fitting:
    @test_logs(
        (:info, r"Feature"),
        fit(
            learner,
            Tables.subset(X, train),
            y[train];
            verbosity=1,
        ),
    )

    # quiet fitting:
    model = @test_logs(
        fit(
            learner,
            Tables.subset(X, train),
            y[train];
            verbosity=0,
        ),
    )

    ŷ = predict(model, Point(), Tables.subset(X, test))
    @test ŷ isa Vector{Float64}
    @test predict(model, Tables.subset(X, test)) == ŷ

    fitobs = LearnAPI.obs(learner, data)
    predictobs = LearnAPI.obs(model, X)
    model = fit(learner, MLUtils.getobs(fitobs, train); verbosity=0)
    @test LearnAPI.target(learner, fitobs) == y
    @test predict(model, Point(), MLUtils.getobs(predictobs, test)) ≈ ŷ
    @test predict(model, LearnAPI.features(learner, fitobs)) ≈ predict(model, X)

    @test LearnAPI.feature_importances(model) isa Vector{<:Pair{Symbol}}

    filename = tempname()
    using Serialization
    small_model = LearnAPI.strip(model)
    serialize(filename, small_model)

    recovered_model = deserialize(filename)
    @test LearnAPI.learner(recovered_model) == learner
    @test predict(
        recovered_model,
        Point(),
        MLUtils.getobs(predictobs, test)
    ) ≈ ŷ

end

# # VARIATION OF RIDGE REGRESSION THAT USES FALLBACK OF LearnAPI.obs

# no docstring here - that goes with the constructor
struct BabyRidge
    lambda::Float64
end


# ## Implementation

"""
    BabyRidge(; lambda=0.1)

Instantiate a ridge regression learner, with regularization of `lambda`.

"""
BabyRidge(; lambda=0.1) = BabyRidge(lambda) # LearnAPI.constructor defined later

struct BabyRidgeFitted{T,F}
    learner::BabyRidge
    coefficients::Vector{T}
    feature_importances::F
end

function LearnAPI.fit(learner::BabyRidge, data; verbosity=1)

    X, y = data

    lambda = learner.lambda
    table = Tables.columntable(X)
    names = Tables.columnnames(table) |> collect
    A = Tables.matrix(table)'

    # apply core learner:
    coefficients = (A*A' + learner.lambda*I)\(A*y) # vector

    feature_importances = nothing

    return BabyRidgeFitted(learner, coefficients, feature_importances)

end

# extracting stuff from training data:
LearnAPI.target(::BabyRidge, data) = last(data)

LearnAPI.learner(model::BabyRidgeFitted) = model.learner

LearnAPI.predict(model::BabyRidgeFitted, ::Point, Xnew) =
    Tables.matrix(Xnew)*model.coefficients

LearnAPI.strip(model::BabyRidgeFitted) =
    BabyRidgeFitted(model.learner, model.coefficients, nothing)

@trait(
    BabyRidge,
    constructor = BabyRidge,
    kinds_of_proxy = (Point(),),
    tags = ("regression",),
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.learner),
        :(LearnAPI.strip),
        :(LearnAPI.obs),
        :(LearnAPI.features),
        :(LearnAPI.target),
        :(LearnAPI.predict),
        :(LearnAPI.feature_importances),
   )
)

# convenience method:
LearnAPI.fit(learner::BabyRidge, X, y; kwargs...) =
    fit(learner, (X, y); kwargs...)


# ## Tests

@testset "test a variation  which does not overload LearnAPI.obs" begin
    learner = BabyRidge(lambda=0.5)

    model = fit(learner, Tables.subset(X, train), y[train]; verbosity=0)
    ŷ = predict(model, Point(), Tables.subset(X, test))
    @test ŷ isa Vector{Float64}

    fitobs = obs(learner, data)
    predictobs = LearnAPI.obs(model, X)
    model = fit(learner, MLUtils.getobs(fitobs, train); verbosity=0)
    @test predict(model, Point(), MLUtils.getobs(predictobs, test)) == ŷ ==
        predict(model, MLUtils.getobs(predictobs, test))
    @test LearnAPI.target(learner, data) == y
    @test LearnAPI.predict(model, X) ≈
        LearnAPI.predict(model, LearnAPI.features(learner, data))
end

true
