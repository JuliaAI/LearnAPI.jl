abstract type MLType end

"""
    LearnAPI.Model

An optional abstract type for models implementing LearnAPI.jl.

If `typeof(m) <: LearnAPI.Model`, then `m` is guaranteed to be a model in the LearnAPI
sense.

!!! warning
    Model types are not required to subtype `LearnAPI.Model`.

# New model implementations

While not a formal requirment, model types implementing the LearAPI interface are
encouraged to subtype `LearnAPI`, unless it is disruptive to do so.

See also [`LearnAPI.functions`](@ref).

"""
abstract type Model <: MLType end

# See src/model_traits.jl for the `methods` trait definition.
