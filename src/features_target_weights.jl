"""
    LearnAPI.target(learner, data) -> target

Return, for each form of `data` supported by the call [`fit(learner, data)`](@ref), the
target part of `data`, in a form suitable for pairing with predictions. The return value
is only meaningful if `learner` is supervised, i.e., if `:(LearnAPI.target) in
LearnAPI.functions(learner)`.

The returned object has the same number of observations
as `data` has and is guaranteed to implement the data interface specified by
[`LearnAPI.data_interface(learner)`](@ref).

# Extended help

## What is a target variable?

Examples of target variables are house prices in real estate pricing estimates, the
"spam"/"not spam" labels in an email spam filtering task, "outlier"/"inlier" labels in
outlier detection, cluster labels in clustering problems, and censored survival times in
survival analysis. For more on targets and target proxies, see the "Reference" section of
the LearnAPI.jl documentation.

## New implementations

The method should be overloaded if [`fit`](@ref) consumes data that includes a target
variable (in the sense above). This will include both [`LearnAPI.Descriminative`](@ref)
and [`LearnAPI.Generative`](@ref) learners, but never [`LearnAPI.Static`](@ref) learners.
Implementation allows for certain meta-functionality, such as cross-validation in
supervised learning, supervised anomaly detection, and density estimation.

If `obs` is being overloaded, then typically it suffices to overload
`LearnAPI.target(learner, observations)` where `observations = obs(learner, data)` and
`data` is any documented supported `data` in calls of the form [`fit(learner,
data)`](@ref), and to then add a declaration of the form

```julia
LearnAPI.target(learner, data) = LearnAPI.target(learner, obs(learner, data))
```
to catch all other forms of supported input `data`.

Remember to ensure the return value of `LearnAPI.target` implements the data
interface specified by [`LearnAPI.data_interface(learner)`](@ref).

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.target)"; overloaded=false))

"""
function target end

"""
    LearnAPI.weights(learner, data) -> weights

Return, for each form of `data` supported by the call [`fit(learner, data)`](@ref), the
per-observation weights part of `data`.

The returned object has the same number of observations as `data` has and is guaranteed to
implement the data interface specified by [`LearnAPI.data_interface(learner)`](@ref).

Where `nothing` is returned, weighting is understood to be uniform.

# Extended help

# New implementations

Implementing is optional.

If `obs` is being overloaded, then typically it suffices to overload
`LearnAPI.weights(learner, observations)` where `observations = obs(learner, data)` and
`data` is any documented supported `data` in calls of the form [`fit(learner,
data)`](@ref), and to add a declaration of the form

```julia
LearnAPI.weights(learner, data) = LearnAPI.weights(learner, obs(learner, data))
```
to catch all other forms of supported input `data`.

Ensure the returned object, unless `nothing`, implements the data interface specified by
[`LearnAPI.data_interface(learner)`](@ref).

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.weights)"; overloaded=false))

"""
weights(::Any, data) = nothing

"""
    LearnAPI.features(learner, data)

Return, for each form of `data` supported by the call [`fit(learner, data)`](@ref), the
features part `X` of `data`.

While "features" will typically have the commonly understood meaning ("covariates" or
"prediuctors"), the only learner-generic guaranteed properties of `X` are:

- `X` can be passed to [`predict`](@ref) or [`transform`](@ref) when these are supported
  by `learner`, as in the call `predict(model, X)`, where `model = fit(learner, data)`.

- `X` has the same number of observations as `data` has and is guaranteed to implement
  the data interface specified by [`LearnAPI.data_interface(learner)`](@ref).

# Extended help

# New implementations

Implementation of this method allows for certain meta-functionality, such as
cross-validation. It can only be implemented for [`LearnAPI.Descriminative`](@ref)
learners.

If `obs` is being overloaded, then typically it suffices to overload
`LearnAPI.features(learner, observations)` where `observations = obs(learner, data)` and
`data` is any documented supported `data` in calls of the form [`fit(learner,
data)`](@ref), and to add a declaration of the form

```julia
LearnAPI.features(learner, data) = LearnAPI.features(learner, obs(learner, data))
```
to catch all other forms of supported input `data`.

Ensure the returned object, implements the data interface specified by
[`LearnAPI.data_interface(learner)`](@ref).

`:(LearnAPI.features)` must be included in the return value of
[`LearnAPI.functions(learner)`](@ref), unless the learner is static (`fit` consumes no
data).

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.target)"; overloaded=false))

"""
function features end
