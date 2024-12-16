# LearnAPI.jl

A base Julia interface for machine learning and statistics

[![Lifecycle:Maturing](https://img.shields.io/badge/Lifecycle-Maturing-007EC6)](ROADMAP.md)
[![Build Status](https://github.com/JuliaAI/LearnAPI.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/LearnAPI.jl/actions)
[![codecov](https://codecov.io/gh/JuliaAI/LearnAPI.jl/graph/badge.svg?token=9IWT9KYINZ)](https://codecov.io/gh/JuliaAI/LearnAPI.jl?branch=dev)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliaai.github.io/LearnAPI.jl/dev/)

Comprehensive documentation is [here](https://juliaai.github.io/LearnAPI.jl/dev/).

New contributions welcome. See the [road map](ROADMAP.md).

## Code snippet

Configure a machine learning algorithm:

```julia
julia> ridge = Ridge(lambda=0.1)
```

Inspect available functionality:

```
julia> @functions ridge
(fit, LearnAPI.learner, LearnAPI.strip, obs, LearnAPI.features, LearnAPI.target, predict, LearnAPI.coefficients
```

Train:

```julia
julia> model = fit(ridge, data)
```

Predict:

```julia
julia> predict(model, data)[1]
"virginica"
```

Predict a probability distribution ([proxy](https://juliaai.github.io/LearnAPI.jl/dev/kinds_of_target_proxy/#proxy_types) for the target):

```julia
julia> predict(model, Distribution(), data)[1]
UnivariateFinite{Multiclass{3}}(setosa=>0.0, versicolor=>0.25, virginica=>0.75)
```

## Credits

Created by Anthony Blaom, in cooperation with Cameron Bieganek and other [members of the
Julia
community](https://discourse.julialang.org/t/ann-learnapi-jl-proposal-for-a-basement-level-machine-learning-api/93048).

