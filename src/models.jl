abstract type MLType end


"""
    LearnAPI.Model

An optional abstract type for models in the ML Model Interface.

# New ML Model Implementations

Either declare `NewModelType <: LearnAPI.Model` or `LearnAPI.model(::SomeModelType) =
true`. The first implies the second and additionally guarantees `==` has correct behaviour
for `NewModelType` instances.

See also [`LearnAPI.ismodel`](@ref).

"""
abstract type Model <: MLType end

"""
    ismodel(m)

Returns `true` exactly when `m` is a *model*, as defined in the ML Model Interface
documentation. In particular, this means:

- `m` is an object whose properties, as returned by `getproperty(m, field)` for `field in
  propertynames(m)`, represent the hyper-parameters of a machine learning algorithm.

- If `n` is another model, then `m == n` if and only if `typeof(n) == typeof(m)` and
  corresponding properties are `==`.

- `m` correctly implements methods from the ML Model Interface. See the documentation for
  LearnAPI for details.


# New ML Model Implementations

Either declare `NewModelType <: LearnAPI.Model` or `LearnAPI.model(::NewModelType) =
true`.

See also [`LearnAPI.Model`](@ref).

"""
ismodel(::Any) = false
ismodel(::Model) = true
