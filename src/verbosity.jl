const DEFAULT_VERBOSITY = Ref(1)

"""
    LearnAPI.default_verbosity()
    LearnAPI.default_verbosity(verbosity::Int)

Respectively return, or set, the default `verbosity` level for LearnAPI.jl methods that
support it, which includes [`fit`](@ref), [`update`](@ref),
[`update_observations`](@ref), and [`update_features`](@ref). The effect in a top-level
call is generally:



| `verbosity` | behaviour     |
|:------------|:--------------|
| 1           | informational |
| 0           | warnings only |


Methods consuming `verbosity` generally call other verbosity-supporting methods
at one level lower, so increasing `verbosity` beyond `1` may be useful.

"""
default_verbosity() = DEFAULT_VERBOSITY[]
default_verbosity(level) = (DEFAULT_VERBOSITY[] = level)
