abstract type LearnAPIType end

"""
    LearnAPI.Algorithm

An optional abstract type for algorithms implementing LearnAPI.jl.

If `typeof(alg) <: LearnAPI.Algorithm`, then `alg` is guaranteed to be an ML/statistical
algorithm in the strict LearnAPI sense.

# New implementations

While not a formal requirement, algorithm types implementing the LearnAPI.jl are
encouraged to subtype `LearnAPI.Algorithm`, unless it is disruptive to do so.

See also [`LearnAPI.functions`](@ref).

"""
abstract type Algorithm <: LearnAPIType end

# See /src/algorithm_traits.jl for the `functions` trait definition.
