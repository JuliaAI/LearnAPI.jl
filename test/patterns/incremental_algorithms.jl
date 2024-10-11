using LearnAPI
using Statistics
using StableRNGs

import Distributions

# # NORMAL DENSITY ESTIMATOR

# An example of density estimation and also of incremental learning
# (`update_observations`).


# ## Implementation

"""
    NormalEstimator()

Instantiate an algorithm for finding the maximum likelihood normal distribution fitting
some real univariate data `y`. Estimates can be updated with new data.

```julia
model = fit(NormalEstimator(), y)
d = predict(model) # returns the learned `Normal` distribution
```

While the above is equivalent to the single operation `d =
predict(NormalEstimator(), y)`, the above workflow allows for the presentation of
additional observations post facto: The following is equivalent to `d2 =
predict(NormalEstimator(), vcat(y, ynew))`:

```julia
update_observations(model, ynew)
d2 = predict(model)
```

Inspect all learned parameters with `LearnAPI.extras(model)`. Predict a 95%
confidence interval with `predict(model, ConfidenceInterval())`

"""
struct NormalEstimator end

struct NormalEstimatorFitted{T}
    Σy::T
    ȳ::T
    ss::T # sum of squared residuals
    n::Int
end

LearnAPI.algorithm(::NormalEstimatorFitted) = NormalEstimator()

function LearnAPI.fit(::NormalEstimator, y)
    n = length(y)
    Σy = sum(y)
    ȳ = Σy/n
    ss = sum(x->x^2, y) - n*ȳ^2
    return NormalEstimatorFitted(Σy, ȳ, ss, n)
end

function LearnAPI.update_observations(model::NormalEstimatorFitted, ynew)
    m = length(ynew)
    n = model.n + m
    Σynew = sum(ynew)
    Σy = model.Σy + Σynew
    ȳ = Σy/n
    δ = model.n*((m*model.ȳ  - Σynew)/n)^2
    ss = model.ss + δ + sum(x -> (x - ȳ)^2, ynew)
    return NormalEstimatorFitted(Σy, ȳ, ss, n)
end

LearnAPI.features(::NormalEstimator, y) = nothing
LearnAPI.target(::NormalEstimator, y) = y

LearnAPI.predict(model::NormalEstimatorFitted, ::Distribution) =
    Distributions.Normal(model.ȳ, sqrt(model.ss/model.n))
LearnAPI.predict(model::NormalEstimatorFitted, ::Point) = model.ȳ
function LearnAPI.predict(model::NormalEstimatorFitted, ::ConfidenceInterval)
    d = predict(model, Distribution())
    return (quantile(d, 0.025), quantile(d, 0.975))
end

# for fit and predict in one line:
LearnAPI.predict(::NormalEstimator, k::LearnAPI.KindOfProxy, y)  =
    predict(fit(NormalEstimator(), y), k)
LearnAPI.predict(::NormalEstimator, y) = predict(NormalEstimator(), Distribution(), y)

LearnAPI.extras(model::NormalEstimatorFitted) = (μ=model.ȳ, σ=sqrt(model.ss/model.n))

@trait(
    NormalEstimator,
    constructor = NormalEstimator,
    kinds_of_proxy = (Distribution(), Point(), ConfidenceInterval()),
    tags = ("density estimation", "incremental algorithms"),
    is_pure_julia = true,
    human_name = "normal distribution estimator",
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.algorithm),
        :(LearnAPI.strip),
        :(LearnAPI.obs),
        :(LearnAPI.features),
        :(LearnAPI.target),
        :(LearnAPI.predict),
        :(LearnAPI.update_observations),
        :(LearnAPI.extras),
    ),
)

# ## Tests

@testset "NormalEstimator" begin
    rng = StableRNG(123)
    y = rand(rng, 50);
    ynew = rand(rng, 10);
    algorithm = NormalEstimator()
    model = fit(algorithm, y)
    d = predict(model)
    μ, σ = Distributions.params(d)
    @test μ ≈ mean(y)
    @test σ ≈ std(y)*sqrt(49/50) # `std` uses Bessel's correction

    # accessor function:
    @test LearnAPI.extras(model) == (; μ, σ)

    # one-liner:
    @test predict(algorithm, y) == d
    @test predict(algorithm, Point(), y) ≈ μ
    @test predict(algorithm, ConfidenceInterval(), y)[1] ≈ quantile(d, 0.025)

    # updating:
    model = update_observations(model, ynew)
    μ2, σ2 = LearnAPI.extras(model)
    μ3, σ3 = LearnAPI.extras(fit(algorithm, vcat(y, ynew))) # training ab initio
    @test μ2 ≈ μ3
    @test σ2 ≈ σ3
end
