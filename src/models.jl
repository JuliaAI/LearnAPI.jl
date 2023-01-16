abstract type MLType end

"""
    LearnAPI.Model

An optional abstract type for models implementing LearnAPI.jl.

If `typeof(m) <: LearnAPI.Model`, then `m` is guaranteed to be a model in the LearnAPI
sense.

# New model implementations

While not a formal requirement, model types implementing the LearnAPI interface are
encouraged to subtype `LearnAPI.Model`, unless it is disruptive to do so.

See also [`LearnAPI.functions`](@ref).

"""
abstract type Model <: MLType end

# See src/model_traits.jl for the `methods` trait definition.
