# LearnAPI.jl

A base Julia interface for machine learning and statistics

[![Lifecycle:Maturing](https://img.shields.io/badge/Lifecycle-Maturing-007EC6)](ROADMAP.md)
[![Build Status](https://github.com/JuliaAI/LearnAPI.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/LearnAPI.jl/actions)
[![codecov](https://codecov.io/gh/JuliaAI/LearnAPI.jl/graph/badge.svg?token=9IWT9KYINZ)](https://codecov.io/gh/JuliaAI/LearnAPI.jl?branch=dev)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliaai.github.io/LearnAPI.jl/dev/)

Comprehensive documentation is [here](https://juliaai.github.io/LearnAPI.jl/dev/).

New contributions welcome. See the [road map](ROADMAP.md).

## Synopsis

Package provides for variations and elaborations on the following basic pattern in machine
learning and statistics:

```julia
model = fit(learner, data)
predict(model, newdata)
```

Here `learner` specifies the configuration the algorithm (the hyperparameters) while
`model` stores learned parameters and any byproducts of algorithm execution.

## Related packages

- [MLCore.jl](https://github.com/JuliaML/MLCore.jl) ([docs](https://juliaml.github.io/MLUtils.jl/stable/api/#Core-API))

- [LearnTestAPI.jl](https://github.com/JuliaAI/LearnTestAPI.jl): Package to test implementations of LearnAPI.jl (but documented here)

- [LearnDataFrontEnds.jl](https://github.com/JuliaAI/LearnDataFrontEnds.jl): for including flexible, user-friendly, data front ends for LearnAPI.jl implementations ([docs]((https://juliaai.github.io/stable/))


## Credits

Created by Anthony Blaom, in cooperation with Cameron Bieganek and other [members of the
Julia
community](https://discourse.julialang.org/t/ann-learnapi-jl-proposal-for-a-basement-level-machine-learning-api/93048).

