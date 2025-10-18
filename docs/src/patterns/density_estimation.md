# Density Estimation

In density estimators, `fit` is trained only on [target data](@ref proxy), and `predict`
consumes no data at all, a pattern flagged by the identities [`LearnAPI.target(learner,
y)`](@ref)` == y` and [`LearnAPI.kind_of(learner)`](@ref)` ==
`[`LearnAPI.Generative()`](@ref), respetively.


Typically `predict` returns a single probability density/mass function. Here's a sample
workflow:

```julia
model = fit(learner, y) # no features
predict(model)          # shortcut for  `predict(model, Distribution())`, or similar
```

A one-line convenience method will typically be implemented as well:

```julia
predict(learner, y)
```

However, having the multi-line workflow enables the possibility of updating the model with
new data. See this example from the JuliaTestAPI.jl test suite:

- [normal distribution estimator](https://github.com/JuliaAI/LearnTestAPI.jl/blob/dev/src/learners/incremental_algorithms.jl)
