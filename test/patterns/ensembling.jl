using LearnAPI
using LinearAlgebra
using Tables
import MLUtils
import DataFrames
using Random
using Statistics
using StableRNGs

# # ENSEMBLE OF REGRESSORS (A MODEL WRAPPER)

# We implement a learner that creates an bagged ensemble of regressors, i.e, where each
# atomic model is trained on a random sample of the training observations (same number,
# but sampled with replacement). In particular this learner has an iteration parameter
# `n`, and we implement `update` to execute a warm restarts when `n` increases.

# no docstring here - that goes with the constructor; some fields left abstract for
# simplicity
#
struct Ensemble
    atom # the base regressor being bagged
    rng
    n::Int
end

# Since the `atom` hyperparameter is another learner, the user must explicitly set it in
# constructor calls or an error is thrown. We also need to overload the
# `LearnAPI.is_composite` trait (done later).

"""
    Ensemble(atom; rng=Random.default_rng(), n=10)

Instantiate a bagged ensemble of `n` regressors, with base regressor `atom`, etc

"""
Ensemble(atom; rng=Random.default_rng(), n=10) =
    Ensemble(atom, rng, n) # `LearnAPI.constructor` defined later

# need a pure keyword argument constructor:
function Ensemble(; atom=nothing, kwargs...)
    isnothing(atom) && error("You must specify `atom=...` ")
    Ensemble(atom; kwargs...)
end

struct EnsembleFitted
    learner::Ensemble
    atom::Ridge
    rng    # mutated copy of `learner.rng`
    models # leaving type abstract for simplicity
end

LearnAPI.learner(model::EnsembleFitted) = model.learner

# We add the same data interface that the atomic regressor uses:
LearnAPI.obs(learner::Ensemble, data) = LearnAPI.obs(learner.atom, data)
LearnAPI.obs(model::EnsembleFitted, data) = LearnAPI.obs(first(model.models), data)
LearnAPI.target(learner::Ensemble, data) = LearnAPI.target(learner.atom, data)
LearnAPI.features(learner::Ensemble, data) = LearnAPI.features(learner.atom, data)

function LearnAPI.fit(learner::Ensemble, data; verbosity=1)

    # unpack hyperparameters:
    atom = learner.atom
    rng = deepcopy(learner.rng) # to prevent mutation of `learner`!
    n = learner.n

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

    return EnsembleFitted(learner, atom, rng, models)

end

# Consistent with the documented `update` contract, we implement this behaviour: If `n` is
# increased, `update` adds new regressors to the ensemble, including any new
# hyperparameter updates (e.g, new `atom`) when computing the new atomic
# models. Otherwise, update is equivalent to retraining from scratch, with the provided
# hyperparameter updates.
function LearnAPI.update(model::EnsembleFitted, data; verbosity=1, replacements...)
    learner_old = LearnAPI.learner(model)
    learner = LearnAPI.clone(learner_old; replacements...)

    :n in keys(replacements) || return fit(learner, data)

    n = learner.n
    Δn = n - learner_old.n
    n < 0 && return fit(model, learner)

    atom = learner.atom
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

    return EnsembleFitted(learner, atom, rng, models)
end

LearnAPI.predict(model::EnsembleFitted, ::Point, data) =
    mean(model.models) do atomic_model
        predict(atomic_model, Point(), data)
    end

LearnAPI.strip(model::EnsembleFitted) = EnsembleFitted(
    model.learner,
    model.atom,
    model.rng,
    LearnAPI.strip.(Ref(model.atom), models),
)

# learner traits (note the inclusion of `iteration_parameter`):
@trait(
    Ensemble,
    constructor = Ensemble,
    iteration_parameter = :n,
    is_composite = true,
    kinds_of_proxy = (Point(),),
    tags = ("regression", "ensemble algorithms", "iterative models"),
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.learner),
        :(LearnAPI.strip),
        :(LearnAPI.obs),
        :(LearnAPI.features),
        :(LearnAPI.target),
        :(LearnAPI.update),
        :(LearnAPI.predict),
   )
)

# convenience method:
LearnAPI.fit(learner::Ensemble, X, y, extras...; kwargs...) =
    fit(learner, (X, y, extras...); kwargs...)
LearnAPI.update(learner::EnsembleFitted, X, y, extras...; kwargs...) =
    update(learner, (X, y, extras...); kwargs...)


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
    atom = Ridge()
    learner = Ensemble(atom; n=4, rng)
    @test LearnAPI.clone(learner) == learner
    @test :(LearnAPI.obs) in LearnAPI.functions(learner)
    @test LearnAPI.target(learner, data) == y
    @test LearnAPI.features(learner, data) == X

    model = @test_logs(
        (:info, r"Trained 4 ridge"),
        fit(learner, Xtrain, y[train]; verbosity=1),
    );

    ŷ4 = predict(model, Point(), Xtest)
    @test ŷ4 == predict(model, Xtest)

    # add 3 atomic models to the ensemble:
    model = update(model, Xtrain, y[train]; verbosity=0, n=7);
    ŷ7 = predict(model, Xtest)

    # compare with cold restart:
    model_cold = fit(LearnAPI.clone(learner; n=7), Xtrain, y[train]; verbosity=0);
    @test ŷ7 ≈ predict(model_cold, Xtest)

    # test that we get a cold restart if another hyperparameter is changed:
    model2 = update(model, Xtrain, y[train]; atom=Ridge(0.05))
    learner2 = Ensemble(Ridge(0.05); n=7, rng)
    model_cold = fit(learner2, Xtrain, y[train]; verbosity=0)
    @test predict(model2, Xtest) ≈ predict(model_cold, Xtest)

end

true
