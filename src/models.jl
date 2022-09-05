abstract type MLType end


"""
    LearnAPI.Model

An optional abstract type for models implementing LearnAPI.jl.


# New model implementations

Either declare `NewModelType <: LearnAPI.Model` or `LearnAPI.model(::SomeModelType) =
true`. The first implies the second and additionally guarantees `==` has correct behaviour
for `NewModelType` instances.

See also [`LearnAPI.ismodel`](@ref).

"""
abstract type Model <: MLType end

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
