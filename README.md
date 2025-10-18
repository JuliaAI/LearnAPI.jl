# LearnAPI.jl

A base Julia interface for machine learning and statistics

[![Lifecycle:Maturing](https://img.shields.io/badge/Lifecycle-Maturing-007EC6)](ROADMAP.md)
[![Build Status](https://github.com/JuliaAI/LearnAPI.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/LearnAPI.jl/actions)
[![codecov](https://codecov.io/gh/JuliaAI/LearnAPI.jl/graph/badge.svg?token=9IWT9KYINZ)](https://codecov.io/gh/JuliaAI/LearnAPI.jl?branch=dev)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliaai.github.io/LearnAPI.jl/dev/)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliaai.github.io/LearnAPI.jl/stable/)

New contributions welcome. See the [road map](ROADMAP.md).

## Synopsis

LearnAPI.jl provides for variations and elaborations on the following basic pattern in machine
learning and statistics:

```julia
model = fit(learner, data)
predict(model, newdata)
```

Here `learner` specifies the configuration the algorithm (the hyperparameters) while
`model` stores learned parameters and any byproducts of algorithm execution.

LearnAPI.jl is mostly method stubs and lots of documentation. It does not provide
meta-algorithms, such as cross-validation, hyperparameter optimization, or model composition, but does aim to
support such algorithms.

## Related packages

- [MLCore.jl](https://github.com/JuliaML/MLCore.jl): The default sub-sampling API (`getobs`/`numbobs`) for LearnAPI.jl implementations, which supports tables and arrays.

- [LearnTestAPI.jl](https://github.com/JuliaAI/LearnTestAPI.jl): Package to test implementations of LearnAPI.jl (but documented here)

- [LearnDataFrontEnds.jl](https://github.com/JuliaAI/LearnDataFrontEnds.jl): For including flexible, user-friendly, data front ends for LearnAPI.jl implementations ([docs](https://juliaai.github.io/LearnDataFrontEnds.jl/stable/))

- [StatisticalMeasures.jl](https://github.com/JuliaAI/StatisticalMeasures.jl): Package providing metrics, compatible with LearnAPI.jl

- [StatsModels.jl](https://github.com/JuliaStats/StatsModels.jl): Provides the R-style formula implementation of data preprocessing handled by [LearnDataFrontEnds.jl](https://github.com/JuliaAI/LearnDataFrontEnds.jl)

### Selected packages providing alternative API's

The following alphabetical list of packages provide public base API's.  Some provide
additional functionality. PR's to add missing items welcome.

- [AutoMLPipeline.jl](https://github.com/IBM/AutoMLPipeline.jl)

- [BetaML.jl](https://github.com/sylvaticus/BetaML.jl)

- [FastAI.jl](https://github.com/FluxML/FastAI.jl) (focused on deep learning)

- [LearnBase.jl](https://github.com/JuliaML/LearnBase.jl) (now archived but of historical interest)

- [MLJModelInterface.jl](https://github.com/JuliaAI/MLJModelInterface.jl)

- [MLUtils.jl](https://github.com/JuliaML/MLUtils.jl) (more than a base API, focused on deep learning)

- [ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl) (an API in addition to being a wrapper for [scikit-learn](https://scikit-learn.org/stable/))

- [StatsAPI.jl](https://github.com/JuliaStats/StatsAPI.jl/tree/main) (specialized to needs of traditional statistical models)


## Credits

Created by Anthony Blaom, in cooperation with Cameron Bieganek and other [members of the
Julia
community](https://discourse.julialang.org/t/ann-learnapi-jl-proposal-for-a-basement-level-machine-learning-api/93048).

