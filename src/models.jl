const DOC_MODEL =
"""

1. The properties of `m`, as returned by `getproperty(m, field)` for `field in propertynames(m)`, represent the hyper-parameters of a machine learning algorithm (reflected in the name of the type of `m`).

2. If `n` is another model, then `m == n` if and only if `typeof(n) == typeof(m)` and corresponding properties are `==`.

"""

abstract type MLType end

"""
    LearnAPI.Model

An optional abstract type for models implementing LearnAPI.jl.

If `typeof(m) <: LearnAPI.Model`, then `m` is guaranteed to be a model in the LearnAPI.jl
sense. Specifically, this means:

$DOC_MODEL

# New model implementations

Overloading [`LearnAPI.functions`](@ref) for a given type implies a promise that instances
of that type are LearnAPI.jl models in the above sense. If one subtypes `LearnAPI.Model`,
then this promise automatically holds true.

See also [`LearnAPI.functions`](@ref).

"""
abstract type Model <: MLType end

# See src/model_traits.jl for the `methods` trait definition.
