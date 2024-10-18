"""
    LearnAPI.clone(learner; replacements...)

Return a shallow copy of `learner` with the specified hyperparameter replacements.

```julia
clone(learner; epochs=100, learning_rate=0.01)
```

It is guaranteed that `LearnAPI.clone(learner) == learner`.

"""
function clone(learner; replacements...)
    reps = NamedTuple(replacements)
    names = propertynames(learner)
    rep_names = keys(reps)

    new_values = map(names) do name
        name in rep_names && return getproperty(reps, name)
        getproperty(learner, name)
    end
    return LearnAPI.constructor(learner)(NamedTuple{names}(new_values)...)
end
