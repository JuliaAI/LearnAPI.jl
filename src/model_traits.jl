"""
    ismodel(m)

Returns `true` exactly when `m` is a *model*, as defined in the LearnAPI.jl
documentation. In particular, this means:

- `m` is an object whose properties, as returned by `getproperty(m, field)` for `field in
  propertynames(m)`, represent the hyper-parameters of a machine learning algorithm.

- If `n` is another model, then `m == n` if and only if `typeof(n) == typeof(m)` and
  corresponding properties are `==`.

- `m` correctly implements zero or more methods from LearnAPI.jl. See the LearnAPI.jl
  documentation for details.


# New model implementations

Either declare `NewModelType <: LearnAPI.Model` or `LearnAPI.model(::NewModelType) =
true`.

See also [`LearnAPI.Model`](@ref).

"""
ismodel(::Any) = false
ismodel(::Model) = true
