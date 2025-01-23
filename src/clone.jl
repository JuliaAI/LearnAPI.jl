"""
    LearnAPI.clone(learner, replacements...)
    LearnAPI.clone(learner; replacements...)

Return a shallow copy of `learner` with the specified hyperparameter replacements. Two
syntaxes are supported, as shown in the following examples:

```julia
clone(learner, :epochs => 100, :learner_rate => 0.01)
clone(learner; epochs=100, learning_rate=0.01)
```

A LearnAPI.jl contract ensures that `LearnAPI.clone(learner) == learner`.

A new learner implementation does not overload `clone`.

"""
function clone(learner, args...; kwargs...)
    reps = merge(NamedTuple(args), NamedTuple(kwargs))
    names = propertynames(learner)
    rep_names = keys(reps)
    new_values = map(names) do name
        name in rep_names && return getproperty(reps, name)
        getproperty(learner, name)
    end
    return LearnAPI.constructor(learner)(NamedTuple{names}(new_values)...)
end
