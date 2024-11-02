const DEFAULT_VERBOSITY = Ref(1)

"""
    LearnAPI.default_verbosity()
    LearnAPI.default_verbosity(level::Int)

Respectively return and set the default verbosity level for LearnAPI.jl, applying, in
particular, to [`fit`](@ref), [`update`](@ref), [`update_observations`](@ref), and
[`update_features`](@ref). The effect in a top-level call is generally:



| `level` | behaviour                         |
|:--------|:----------------------------------|
| 1       | informational                     |
| 0       | warnings only                     |
| -1      | silent                            |


Methods consuming `verbosity` generally call other verbosity-supporting methods
at one level lower, so increasing `verbosity` beyond `1` may be useful.

"""
default_verbosity() = DEFAULT_VERBOSITY[]
default_verbosity(level) = (DEFAULT_VERBOSITY[] = level)
