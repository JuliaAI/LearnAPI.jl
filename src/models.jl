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

# See src/model_traits.jl for the `ismodel` trait definition.
