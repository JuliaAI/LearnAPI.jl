# LearnAPI.jl

A base Julia interface for machine learning and statistics

[![Lifecycle:Maturing](https://img.shields.io/badge/Lifecycle-Maturing-007EC6)](ROADMAP.md)
[![Build Status](https://github.com/JuliaAI/LearnAPI.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/LearnAPI.jl/actions)
[![Coverage](https://codecov.io/gh/JuliaAI/LearnAPI.jl/branch/master/graph/badge.svg)](https://codecov.io/github/JuliaAI/LearnAPI.jl?branch=master)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliaai.github.io/LearnAPI.jl/dev/)

Comprehensive documentation is [here](https://juliaai.github.io/LearnAPI.jl/dev/).

New contributions welcome. See the [road map](ROADMAP.md).

## Code snippet

Configure a learning algorithm, and inspect available functionality:

```julia
julia> algorithm = Ridge(lambda=0.1)
julia> LearnAPI.functions(algorithm)
(:(LearnAPI.fit), :(LearnAPI.algorithm), :(LearnAPI.minimize), :(LearnAPI.obs), 
:(LearnAPI.features), :(LearnAPI.target), :(LearnAPI.predict), :(LearnAPI.coefficients))
```

Train:

```julia
julia> model = fit(algorithm, data)
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

Created by Anthony Blaom, in cooperation with [members of the Julia
community](https://discourse.julialang.org/t/ann-learnapi-jl-proposal-for-a-basement-level-machine-learning-api/93048).

