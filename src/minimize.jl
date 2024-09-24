"""
    minimize(model; options...)

Return a version of `model` that will generally have a smaller memory allocation than
`model`, suitable for serialization. Here `model` is any object returned by
[`fit`](@ref). Accessor functions that can be called on `model` may not work on
`minimize(model)`, but [`predict`](@ref), [`transform`](@ref) and
[`inverse_transform`](@ref) will work, if implemented. Check
`LearnAPI.functions(LearnAPI.algorithm(model))` to view see what the original `model`
implements.

Specific algorithms may provide keyword `options` to control how much of the original
functionality is preserved by `minimize`.

# Extended help

# New implementations

Overloading `minimize` for new algorithms is optional. The fallback is the
identity. $(DOC_IMPLEMENTED_METHODS(":minimize", overloaded=true))

New implementations must enforce the following identities, whenever the right-hand side is
defined:

```julia
predict(minimize(model; options...), args...; kwargs...) ==
    predict(model, args...; kwargs...)
transform(minimize(model; options...), args...; kwargs...) ==
    transform(model, args...; kwargs...)
inverse_transform(minimize(model; options), args...; kwargs...) ==
    inverse_transform(model, args...; kwargs...)
```

Additionally:

```julia
minimize(minimize(model)) == minimize(model)
```

"""
minimize(model) = model
