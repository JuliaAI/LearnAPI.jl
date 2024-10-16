using Pkg
Pkg.activate("perceptron", shared=true)

using LearnAPI
using Random
using Statistics
using StableRNGs
import Optimisers
import Zygote
import NNlib
import CategoricalDistributions
import CategoricalDistributions: pdf, mode
import ComponentArrays

# # PERCEPTRON

# We implement a simple perceptron classifier to illustrate some common patterns for
# gradient descent algorithms. This includes implementation of the following methods:

# - `update`
# - `update_observations`
# - `iteration_parameter`
# - `training_losses`
# - `obs` for pre-processing (non-tabular) classification training data
# - `predict(learner, ::Distribution, Xnew)`

# For simplicity, we use single-observation batches for gradient descent updates, and we
# may dodge some optimizations.

# This is an example of a probability-predicting classifier.


# ## Helpers

"""
    brier_loss(probs, hot)

Return Brier (quadratic) loss.

- `probs`: predicted probability vector
- `hot`: corresponding ground truth observation, as a one-hot encoded `BitVector`

"""
function brier_loss(probs, hot)
    offset = 1 + sum(probs.^2)
    return offset - 2*(sum(probs.*hot))
end

"""
    corefit(perceptron, optimiser, X, y_hot, epochs, state, verbosity)

Return updated `perceptron`, `state` and training losses by carrying out gradient descent
for the specified number of `epochs`.

- `perceptron`: component array with components `weights` and `bias`
- `optimiser`: optimiser from Optimiser.jl
- `X`: feature matrix, of size `(p, n)`
- `y_hot`: one-hot encoded target, of size `(nclasses, n)`
- `epochs`: number of epochs
- `state`: optimiser state

"""
function corefit(perceptron, X, y_hot, epochs, state, verbosity)
    n = size(y_hot) |> last
    losses = map(1:epochs) do _
        total_loss = zero(Float32)
        for i in 1:n
            loss, grad = Zygote.withgradient(perceptron) do p
                probs = p.weights*X[:,i] + p.bias |> NNlib.softmax
                brier_loss(probs, y_hot[:,i])
            end
            ∇loss = only(grad)
            state, perceptron = Optimisers.update(state, perceptron, ∇loss)
            total_loss += loss
        end
        # make some noise, if allowed:
        verbosity > 0 && @info "Training loss: $total_loss"
        total_loss
    end
    return perceptron, state, losses
end


# ## Implementation

# ### Learner

# no docstring here - that goes with the constructor;
# SOME FIELDS LEFT ABSTRACT FOR SIMPLICITY
struct PerceptronClassifier
    epochs::Int
    optimiser # an optmiser from Optimsers.jl
    rng
end

"""
    PerceptronClassifier(; epochs=50, optimiser=Optimisers.Adam(), rng=Random.default_rng())

Instantiate a perceptron classifier.

Train an instance, `learner`, by doing `model = fit(learner, X, y)`, where

-  `X is a `Float32` matrix, with observations-as-columns
-  `y` (target) is some one-dimensional `CategoricalArray`.

Get probabilistic predictions with `predict(model, Xnew)` and
point predictions with `predict(model, Point(), Xnew)`.

# Warm restart options

    update_observations(model, newdata; replacements...)

Return an updated model, with the weights and bias of the previously learned perceptron
used as the starting state in new gradient descent updates. Adopt any specified
hyperparameter `replacements` (properties of `LearnAPI.learner(model)`).

    update(model, newdata; epochs=n, replacements...)

If `Δepochs = n - perceptron.epochs` is non-negative, then return an updated model, with
the weights and bias of the previously learned perceptron used as the starting state in
new gradient descent updates for `Δepochs` epochs, and using the provided `newdata`
instead of the previous training data. Any other hyperparaameter `replacements` are also
adopted. If `Δepochs` is negative or not specified, instead return `fit(learner,
newdata)`, where `learner=LearnAPI.clone(learner; epochs=n, replacements....)`.

"""
PerceptronClassifier(; epochs=50, optimiser=Optimisers.Adam(), rng=Random.default_rng()) =
    PerceptronClassifier(epochs, optimiser, rng)


# ### Data interface

# For raw training data:
LearnAPI.target(learner::PerceptronClassifier, data::Tuple) = last(data)

# For wrapping pre-processed training data (output of `obs(learner, data)`):
struct PerceptronClassifierObs
    X::Matrix{Float32}
    y_hot::BitMatrix  # one-hot encoded target
    classes           # the (ordered) pool of `y`, as `CategoricalValue`s
end

# For pre-processing the training data:
function LearnAPI.obs(learner::PerceptronClassifier, data::Tuple)
    X, y = data
    classes = CategoricalDistributions.classes(y)
    y_hot = classes .== permutedims(y) # one-hot encoding
    return PerceptronClassifierObs(X, y_hot, classes)
end

# implement `RadomAccess()` interface for output of `obs`:
Base.length(observations::PerceptronClassifierObs) = length(observations.y)
Base.getindex(observations, I) = PerceptronClassifierObs(
    (@view observations.X[:, I]),
    (@view observations.y[I]),
    observations.classes,
)

LearnAPI.target(
    learner::PerceptronClassifier,
    observations::PerceptronClassifierObs,
) = observations.y

LearnAPI.features(
    learner::PerceptronClassifier,
    observations::PerceptronClassifierObs,
) = observations.X

# Note that data consumed by `predict` needs no pre-processing, so no need to overload
# `obs(model, data)`.


# ### Fitting and updating

# For wrapping outcomes of learning:
struct PerceptronClassifierFitted
    learner::PerceptronClassifier
    perceptron  # component array storing weights and bias
    state       # optimiser state
    classes     # target classes
    losses
end

LearnAPI.learner(model::PerceptronClassifierFitted) = model.learner

# `fit` for pre-processed data (output of `obs(learner, data)`):
function LearnAPI.fit(
    learner::PerceptronClassifier,
    observations::PerceptronClassifierObs;
    verbosity=1,
    )

    # unpack hyperparameters:
    epochs = learner.epochs
    optimiser = learner.optimiser
    rng = deepcopy(learner.rng) # to prevent mutation of `learner`!

    # unpack data:
    X = observations.X
    y_hot = observations.y_hot
    classes = observations.classes
    nclasses = length(classes)

    # initialize bias and weights:
    weights = randn(rng, Float32, nclasses, p)
    bias = zeros(Float32, nclasses)
    perceptron = (; weights, bias) |> ComponentArrays.ComponentArray

    # initialize optimiser:
    state = Optimisers.setup(optimiser, perceptron)

    perceptron, state, losses = corefit(perceptron, X, y_hot, epochs, state, verbosity)

    return PerceptronClassifierFitted(learner, perceptron, state, classes, losses)
end

# `fit` for unprocessed data:
LearnAPI.fit(learner::PerceptronClassifier, data; kwargs...) =
    fit(learner, obs(learner, data); kwargs...)

# see the `PerceptronClassifier` docstring for `update_observations` logic.
function LearnAPI.update_observations(
    model::PerceptronClassifierFitted,
    observations_new::PerceptronClassifierObs;
    verbosity=1,
    replacements...,
    )

    # unpack data:
    X = observations_new.X
    y_hot = observations_new.y_hot
    classes = observations_new.classes
    nclasses = length(classes)

    classes == model.classes || error("New training target has incompatible classes.")

    learner_old = LearnAPI.learner(model)
    learner = LearnAPI.clone(learner_old; replacements...)

    perceptron = model.perceptron
    state = model.state
    losses = model.losses
    epochs = learner.epochs

    perceptron, state, losses_new = corefit(perceptron, X, y_hot, epochs, state, verbosity)
    losses = vcat(losses, losses_new)

    return PerceptronClassifierFitted(learner, perceptron, state, classes, losses)
end
LearnAPI.update_observations(model::PerceptronClassifierFitted, data; kwargs...) =
    update_observations(model, obs(LearnAPI.learner(model), data); kwargs...)

# see the `PerceptronClassifier` docstring for `update` logic.
function LearnAPI.update(
    model::PerceptronClassifierFitted,
    observations::PerceptronClassifierObs;
    verbosity=1,
    replacements...,
    )

    # unpack data:
    X = observations.X
    y_hot = observations.y_hot
    classes = observations.classes
    nclasses = length(classes)

    classes == model.classes || error("New training target has incompatible classes.")

    learner_old = LearnAPI.learner(model)
    learner = LearnAPI.clone(learner_old; replacements...)
    :epochs in keys(replacements) || return fit(learner, observations)

    perceptron = model.perceptron
    state = model.state
    losses = model.losses

    epochs = learner.epochs
    Δepochs = epochs - learner_old.epochs
    epochs < 0 && return fit(model, learner)

    perceptron, state, losses_new = corefit(perceptron, X, y_hot, Δepochs, state, verbosity)
    losses = vcat(losses, losses_new)

    return PerceptronClassifierFitted(learner, perceptron, state, classes, losses)
end
LearnAPI.update(model::PerceptronClassifierFitted, data; kwargs...) =
    update(model, obs(LearnAPI.learner(model), data); kwargs...)


# ### Predict

function LearnAPI.predict(model::PerceptronClassifierFitted, ::Distribution, Xnew)
    perceptron = model.perceptron
    classes = model.classes
    probs = perceptron.weights*Xnew .+ perceptron.bias |> NNlib.softmax
    return CategoricalDistributions.UnivariateFinite(classes, probs')
end

LearnAPI.predict(model::PerceptronClassifierFitted, ::Point, Xnew) =
    mode.(predict(model, Distribution(), Xnew))


# ### Accessor functions

LearnAPI.training_losses(model::PerceptronClassifierFitted) = model.losses


# ### Traits

@trait(
    PerceptronClassifier,
    constructor = PerceptronClassifier,
    iteration_parameter = :epochs,
    kinds_of_proxy = (Distribution(), Point()),
    tags = ("classification", "iterative algorithms", "incremental algorithms"),
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.learner),
        :(LearnAPI.strip),
        :(LearnAPI.obs),
        :(LearnAPI.features),
        :(LearnAPI.target),
        :(LearnAPI.update),
        :(LearnAPI.update_observations),
        :(LearnAPI.predict),
        :(LearnAPI.training_losses),
   )
)


# ### Convenience methods

LearnAPI.fit(learner::PerceptronClassifier, X, y; kwargs...) =
    fit(learner, (X, y); kwargs...)
LearnAPI.update_observations(learner::PerceptronClassifier, X, y; kwargs...) =
    update_observations(learner, (X, y); kwargs...)
LearnAPI.update(learner::PerceptronClassifier, X, y; kwargs...) =
    update(learner, (X, y); kwargs...)


# ## Tests

# synthetic test data:
N = 10
n = 10N # number of observations
p = 2   # number of features
train = 1:6N
test = (6N+1:10N)
rng = StableRNG(123)
X = randn(rng, Float32, p, n);
coefficients = rand(rng, Float32, p)'
y_continuous = coefficients*X |> vec
η1 = quantile(y_continuous, 1/3)
η2 = quantile(y_continuous, 2/3)
y = map(y_continuous) do η
    η < η1 && return "A"
    η < η2 && return "B"
    "C"
end |> CategoricalDistributions.categorical;
Xtrain = X[:, train];
Xtest = X[:, test];
ytrain = y[train];
ytest = y[test];

@testset "PerceptronClassfier" begin
    rng = StableRNG(123)
    learner = PerceptronClassifier(; optimiser=Optimisers.Adam(0.01), epochs=40, rng)
    @test LearnAPI.clone(learner) == learner
    @test :(LearnAPI.update) in LearnAPI.functions(learner)
    @test LearnAPI.target(learner, (X, y)) == y
    @test LearnAPI.features(learner, (X, y)) == X

    model40 = fit(learner, Xtrain, ytrain; verbosity=0)

    # 40 epochs is sufficient for 90% accuracy in this case:
    @test sum(predict(model40, Point(), Xtest) .== ytest)/length(ytest) > 0.9

    # get probabilistic predictions:
    ŷ40 = predict(model40, Distribution(), Xtest);
    @test predict(model40, Xtest) ≈ ŷ40

    # add 30 epochs in an `update`:
    model70 = update(model40, Xtrain, y[train]; verbosity=0, epochs=70)
    ŷ70 = predict(model70, Xtest);
    @test !(ŷ70 ≈ ŷ40)

    # compare with cold restart:
    model = fit(LearnAPI.clone(learner; epochs=70), Xtrain, y[train]; verbosity=0);
    @test ŷ70 ≈ predict(model, Xtest)

    # instead add 30 epochs using `update_observations` instead:
    model70b = update_observations(model40, Xtrain, y[train]; verbosity=0, epochs=30)
    @test ŷ70 ≈ predict(model70b, Xtest) ≈ predict(model, Xtest)
end

true
