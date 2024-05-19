# # DOC STRING HELPERS

const TRAINING_FUNCTIONS = (:fit,)


# # FIT

"""
    LearnAPI.fit(algorithm, data; verbosity=1)
    LearnAPI.fit(algorithm; verbosity=1)

Execute the algorithm with configuration `algorithm` using the provided training `data`,
returning an object, `model`, on which other methods, such as [`predict`](@ref) or
[`transform`](@ref), can be dispatched.  [`LearnAPI.functions(algorithm)`](@ref) returns a
list of methods that can be applied to either `algorithm` or `model`.

The second signature applies to algorithms which do not generalize to new observations. In
that case `predict` or `transform` actually execute the algorithm, but may also write to
the (mutable) object returned by `fit`.

When `data` is a tuple, a data slurping form of `fit` is typically provided.

```julia
model = fit(algorithm, (X, y))  # or `fit(algorithm, X, y)`
ŷ = predict(model, X)
```

Use `verbosity=0` for warnings only, and `-1` for silent training.

See also [`predict`](@ref), [`transform`](@ref), [`inverse_transform`](@ref),
[`LearnAPI.functions`](@ref), [`obs`](@ref).

# Extended help

# New implementations

Implementation is compulsory. The signature must include `verbosity`.

"""
fit(algorithm, data...; kwargs...) = nothing
