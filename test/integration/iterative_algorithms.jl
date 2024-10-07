using LearnAPI
using LinearAlgebra
using Tables
import MLUtils
import DataFrames
using Random
using Statistics
using StableRNGs

# # ENSEMBLE OF RIDGE REGRESSORS

# We implement a toy algorithm that creates an bagged ensemble of ridge regressors (as
# defined already in test/integration/regressors.jl), i.e, where each atomic model is
# trained on a random sample of the training observations (same number, but sampled with
# replacement). In particular this algorithm has an iteration parameter `n`, and we
# implement `update` for warm restarts when `n` increases.

# By re-using the data interface for `Ridge`, we ensure that the resampling (bagging) is
# more efficient (no repeated table -> matrix conversions, and we resample matrices
# directly, not the tables). 

# no docstring here - that goes with the constructor
struct RidgeEnsemble
    lambda::Float64
    rng # leaving abstract for simplicity
    n::Int
end

"""
    RidgeEnsemble(; lambda=0.1, rng=Random.default_rng(), n=10)

Instantiate a RidgeEnsemble algorithm, bla, bla, bla...

"""
RidgeEnsemble(; lambda=0.1, rng=Random.default_rng(), n=10) =
    RidgeEnsemble(lambda, rng, n) # LearnAPI.constructor defined later

struct RidgeEnsembleFitted
    algorithm::RidgeEnsemble
    atom::Ridge
    rng    # mutated copy of `algorithm.rng`
    models # leaving type abstract for simplicity
end

LearnAPI.algorithm(model::RidgeEnsembleFitted) = model.algorithm

# we use the same data interface we provided for `Ridge` in regression.jl:
LearnAPI.obs(algorithm::RidgeEnsemble, data) = LearnAPI.obs(Ridge(), data)
LearnAPI.obs(model::RidgeEnsembleFitted, data) = LearnAPI.obs(first(model.models), data)
LearnAPI.target(algorithm::RidgeEnsemble, data) = LearnAPI.target(Ridge(), data)
LearnAPI.features(algorithm::Ridge, data) = LearnAPI.features(Ridge(), data)

function d(rng)
    i = digits(rng.state)
    m = min(length(i), 4)
    tail = i[end - m + 1:end]
    println(join(string.(tail)))
end

# because we need observation subsampling, we first implement `fit` for output of
# `obs`:
function LearnAPI.fit(algorithm::RidgeEnsemble, data::RidgeFitObs; verbosity=1)

    # unpack hyperparameters:
    lambda = algorithm.lambda
    rng = deepcopy(algorithm.rng) # to prevent mutation of `algorithm`
    n = algorithm.n

    # instantiate atomic algorithm:
    atom = Ridge(lambda)

    # initialize ensemble:
    models = []

    # get number of observations:
    N = MLUtils.numobs(data)

    # train the ensemble:
    for _ in 1:n
        bag = rand(rng, 1:N, N)
        data_subset = MLUtils.getobs(data, bag)
        # step down one verbosity level in atomic fit:
        model = fit(atom, data_subset; verbosity=verbosity - 1)
        push!(models, model)
    end

    # make some noise, if allowed:
    verbosity > 0 && @info "Trained $n ridge regression models. "

    return RidgeEnsembleFitted(algorithm, atom, rng, models)

end

# ... and so need a `fit` for unprocessed `data = (X, y)`:
LearnAPI.fit(algorithm::RidgeEnsemble, data; kwargs...) =
    fit(algorithm, obs(algorithm, data); kwargs...)

# If `n` is increased, this `update` adds new regressors to the ensemble, including any
# new # hyperparameter updates (e.g, `lambda`) when computing the new
# regressors. Otherwise, update is equivalent to retraining from scratch, with the
# provided hyperparameter updates.
function LearnAPI.update(
    model::RidgeEnsembleFitted,
    data::RidgeFitObs;
    verbosity=1,
    replacements...,
    )

    :n in keys(replacements) || return fit(model, data)

    algorithm_old = LearnAPI.algorithm(model)
    algorithm = LearnAPI.clone(algorithm_old; replacements...)
    n = algorithm.n
    Δn = n - algorithm_old.n
    n < 0 && return fit(model, algorithm)

    # get number of observations:
    N = MLUtils.numobs(data)

    # initialize:
    models = model.models
    rng = model.rng # as mutated in previous `fit`/`update` calls

    atom = Ridge(; lambda=algorithm.lambda)

    rng2 = StableRNG(123)
    for _ in 1:10
        rand(rng2)
    end

    # add new regressors to the ensemble:
    for _ in 1:Δn
        bag = rand(rng, 1:N, N)
        data_subset = MLUtils.getobs(data, bag)
        model = fit(atom, data_subset; verbosity=verbosity-1)
        push!(models, model)
    end

    # make some noise, if allowed:
    verbosity > 0 && @info "Trained $Δn additional ridge regression models. "

    return RidgeEnsembleFitted(algorithm, atom, rng, models)
end

# an `update` for unprocessed `data = (X, y)`:
LearnAPI.update(model::RidgeEnsembleFitted, data; kwargs...) =
    update(model, obs(LearnAPI.algorithm(model), data); kwargs...)

# `data` here can be pre-processed or not, because we're just calling the atomic
# `predict`, which already has a data interface, and we don't need any subsampling, like
# we did for `fit`:
LearnAPI.predict(model::RidgeEnsembleFitted, ::Point, data) =
    mean(model.models) do atomic_model
        predict(atomic_model, Point(), data)
    end

LearnAPI.minimize(model::RidgeEnsembleFitted) = RidgeEnsembleFitted(
    model.algorithm,
    model.atom,
    model.rng,
    minimize.(Ref(model.atom), models),
)

# note the inclusion of `iteration_parameter`:
@trait(
    RidgeEnsemble,
    constructor = RidgeEnsemble,
    iteration_parameter = :n,
    kinds_of_proxy = (Point(),),
    tags = ("regression", "ensemble algorithms", "iterative models"),
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.algorithm),
        :(LearnAPI.minimize),
        :(LearnAPI.obs),
        :(LearnAPI.features),
        :(LearnAPI.target),
        :(LearnAPI.update),
        :(LearnAPI.predict),
        :(LearnAPI.feature_importances),
   )
)

# synthetic test data:
N = 10 # number of observations
train = 1:6
test = 7:10
a, b, c = rand(N), rand(N), rand(N)
X = (; a, b, c)
X = DataFrames.DataFrame(X)
y = 2a - b + 3c + 0.05*rand(N)
data = (X, y)
Xtrain = Tables.subset(X, train)
Xtest = Tables.subset(X, test)

@testset "test an implementation of bagged ensemble of ridge regressors" begin
    rng = StableRNG(123)
    algorithm = RidgeEnsemble(lambda=0.5, n=4; rng)
    @test LearnAPI.clone(algorithm) == algorithm
    @test :(LearnAPI.obs) in LearnAPI.functions(algorithm)
    @test LearnAPI.target(algorithm, data) == y
    @test LearnAPI.features(algorithm, data) == X

    model = @test_logs(
        (:info, r"Trained 4 ridge"),
        fit(algorithm, Xtrain, y[train]; verbosity=1),
    );

    ŷ4 = predict(model, Point(), Xtest)
    @test ŷ4 == predict(model, Xtest)

    # add 3 atomic models to the ensemble:
    # model = @test_logs(
    #     (:info, r"Trained 3 additional"),
    #     update(model, Xtrain, y[train]; n=7),
    # )
    model = update(model, Xtrain, y[train]; verbosity=0, n=7);
    ŷ7 = predict(model, Xtest)

    # compare with cold restart:
    model = fit(LearnAPI.clone(algorithm; n=7), Xtrain, y[train]; verbosity=0);
    @test ŷ7 ≈ predict(model, Xtest)


    update(model, Xtest;
    fitobs = LearnAPI.obs(algorithm, data)
    predictobs = LearnAPI.obs(model, X)
    model = fit(algorithm, MLUtils.getobs(fitobs, train); verbosity=0)
    @test LearnAPI.target(algorithm, fitobs) == y
    @test predict(model, Point(), MLUtils.getobs(predictobs, test)) ≈ ŷ
    @test predict(model, LearnAPI.features(algorithm, fitobs)) ≈ predict(model, X)

    @test LearnAPI.feature_importances(model) isa Vector{<:Pair{Symbol}}

    filename = tempname()
    using Serialization
    small_model = minimize(model)
    serialize(filename, small_model)

    recovered_model = deserialize(filename)
    @test LearnAPI.algorithm(recovered_model) == algorithm
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

LearnAPI.predict(model::BabyRidgeFitted, ::Point, Xnew) =
    Tables.matrix(Xnew)*model.coefficients

LearnAPI.minimize(model::BabyRidgeFitted) =
    BabyRidgeFitted(model.algorithm, model.coefficients, nothing)

@trait(
    BabyRidge,
    constructor = BabyRidge,
    kinds_of_proxy = (Point(),),
    tags = ("regression",),
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.algorithm),
        :(LearnAPI.minimize),
        :(LearnAPI.obs),
        :(LearnAPI.features),
        :(LearnAPI.target),
        :(LearnAPI.predict),
        :(LearnAPI.feature_importances),
   )
)

@testset "test a variation  which does not overload LearnAPI.obs" begin
           algorithm = BabyRidge(lambda=0.5)
           @test

    model = fit(algorithm, Tables.subset(X, train), y[train]; verbosity=0)
    ŷ = predict(model, Point(), Tables.subset(X, test))
    @test ŷ isa Vector{Float64}

    fitobs = obs(algorithm, data)
    predictobs = LearnAPI.obs(model, X)
    model = fit(algorithm, MLUtils.getobs(fitobs, train); verbosity=0)
    @test predict(model, Point(), MLUtils.getobs(predictobs, test)) == ŷ ==
        predict(model, MLUtils.getobs(predictobs, test))
    @test LearnAPI.target(algorithm, data) == y
    @test LearnAPI.predict(model, X) ≈
        LearnAPI.predict(model, LearnAPI.features(algorithm, data))
end

true
