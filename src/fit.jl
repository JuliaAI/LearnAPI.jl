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

The second signature is provided by algorithms that do not generalize to new observations
("static" algorithms). In that case, `transform(model, data)` or `predict(model, ...,
data)` carries out the actual algorithm execution, writing any byproducts of that
operation to the mutable object `model` returned by `fit`.

Whenever `fit` expects a tuple form of argument, `data = (X1, ..., Xn)`, then the
signature `fit(algorithm, X1, ..., Xn)` is also provided.

For example, a supervised classifier will typically admit this workflow:

```julia
model = fit(algorithm, (X, y)) # or `fit(algorithm, X, y)`
ŷ = predict(model, Xnew)
```

Use `verbosity=0` for warnings only, and `-1` for silent training.

See also [`predict`](@ref), [`transform`](@ref), [`inverse_transform`](@ref),
[`LearnAPI.functions`](@ref), [`obs`](@ref).

# Extended help

# New implementations

Implementation is compulsory. The signature must include `verbosity`. Note the requirement
on providing slurping signatures. A fallback for the first signature calls the second,
ignoring `data`:

```julia
fit(algorithm, data; kwargs...) = fit(algorithm; kwargs...)
```
$(DOC_DATA_INTERFACE(:fit))

"""
fit(algorithm, data; kwargs...) =
    fit(algorithm; kwargs...)
fit(algorithm, data1, datas...; kwargs...) =
    fit(algorithm, (data1, datas...); kwargs...)
