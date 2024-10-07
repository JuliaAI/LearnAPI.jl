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

# We add the same data interface we provided for `Ridge` in regression.jl. This is an
# optional step on which the later code does not depend.
LearnAPI.obs(algorithm::RidgeEnsemble, data) = LearnAPI.obs(Ridge(), data)
LearnAPI.obs(model::RidgeEnsembleFitted, data) = LearnAPI.obs(first(model.models), data)
LearnAPI.target(algorithm::RidgeEnsemble, data) = LearnAPI.target(Ridge(), data)
LearnAPI.features(algorithm::Ridge, data) = LearnAPI.features(Ridge(), data)

function LearnAPI.fit(algorithm::RidgeEnsemble, data; verbosity=1)

    # unpack hyperparameters:
    lambda = algorithm.lambda
    rng = deepcopy(algorithm.rng) # to prevent mutation of `algorithm`
    n = algorithm.n

    # instantiate atomic algorithm:
    atom = Ridge(lambda)

    # ensure data can be subsampled using MLUtils.jl, and that we're feeding the atomic
    # `fit` data in an efficient (pre-processed) form:

    observations = obs(atom, data)

    # initialize ensemble:
    models = []

    # get number of observations:
    N = MLUtils.numobs(observations)

    # train the ensemble:
    for _ in 1:n
        bag = rand(rng, 1:N, N)
        data_subset = MLUtils.getobs(observations, bag)
        # step down one verbosity level in atomic fit:
        model = fit(atom, data_subset; verbosity=verbosity - 1)
        push!(models, model)
    end

    # make some noise, if allowed:
    verbosity > 0 && @info "Trained $n ridge regression models. "

    return RidgeEnsembleFitted(algorithm, atom, rng, models)

end

# If `n` is increased, this `update` adds new regressors to the ensemble, including any
# new # hyperparameter updates (e.g, `lambda`) when computing the new
# regressors. Otherwise, update is equivalent to retraining from scratch, with the
# provided hyperparameter updates.
function LearnAPI.update(model::RidgeEnsembleFitted, data; verbosity=1, replacements...)
    :n in keys(replacements) || return fit(model, data)

    algorithm_old = LearnAPI.algorithm(model)
    algorithm = LearnAPI.clone(algorithm_old; replacements...)
    n = algorithm.n
    Δn = n - algorithm_old.n
    n < 0 && return fit(model, algorithm)

    atom = Ridge(; lambda=algorithm.lambda)
    observations = obs(atom, data)
    N = MLUtils.numobs(observations)

    # initialize:
    models = model.models
    rng = model.rng # as mutated in previous `fit`/`update` calls

    # add new regressors to the ensemble:
    for _ in 1:Δn
        bag = rand(rng, 1:N, N)
        data_subset = MLUtils.getobs(observations, bag)
        model = fit(atom, data_subset; verbosity=verbosity-1)
        push!(models, model)
    end

    # make some noise, if allowed:
    verbosity > 0 && @info "Trained $Δn additional ridge regression models. "

    return RidgeEnsembleFitted(algorithm, atom, rng, models)
end

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

end

true
