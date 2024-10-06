"""
    LearnAPI.clone(algorithm; replacements...)

Return a shallow copy of `algorithm` with the specified hyperparameter replacements.

```julia
clone(algorithm; epochs=100, learning_rate=0.01)
```

It is guaranted that `LearnAPI.clone(algorithm) == algorithm`.

"""
function clone(algorithm; replacements...)
    reps = NamedTuple(replacements)
    names = propertynames(algorithm)
    rep_names = keys(reps)

    new_values = map(names) do name
        name in rep_names && return getproperty(reps, name)
        getproperty(algorithm, name)
    end
    return LearnAPI.constructor(algorithm)(NamedTuple{names}(new_values)...)
end
