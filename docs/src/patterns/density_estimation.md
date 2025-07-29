# Density Estimation

In density estimators, `fit` is trained only on [target data](@ref proxy), and
`predict(model, kind_of_proxy)` consumes no data at all. Typically `predict` returns a
single probability density/mass function (`kind_of_proxy = Distribution()`). 

Here's a sample workflow:

```julia
model = fit(learner, y) # no features
predict(model)  # shortcut for  `predict(model, SingleDistribution())`, or similar
```

A one-liner will typically be implemented as well:

```julia
predict(learner, y)
```

A density estimator, `learner`, will need to arrange that
[`LearnAPI.features(learner, data)`](@ref) always returns `nothing` and
[`LearnAPI.sees_features(learner)`](@ref) returns `false`.

See these examples from the JuliaTestAPI.jl test suite:

- [normal distribution estimator](https://github.com/JuliaAI/LearnTestAPI.jl/blob/dev/src/learners/incremental_algorithms.jl)
