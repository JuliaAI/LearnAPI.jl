abstract type MLType end

"""
    LearnAPI.Algorithm

An optional abstract type for algorithms implementing LearnAPI.jl.

If `typeof(alg) <: LearnAPI.Algorithm`, then `alg` is guaranteed to be an algorithm in the
LearnAPI sense.

# New implementations

While not a formal requirement, algorithm types implementing LearnAPI are encouraged to
subtype `LearnAPI.Algorithm`, unless it is disruptive to do so.

See also [`LearnAPI.functions`](@ref).

"""
abstract type Algorithm <: MLType end

# See src/algorithm_traits.jl for the `methods` trait definition.
