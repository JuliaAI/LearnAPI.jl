# # FIT

"""
    fit(learner, data; verbosity=1)
    fit(learner; verbosity=1)

Execute the machine learning or statistical algorithm with configuration `learner` using
the provided training `data`, returning an object, `model`, on which other methods, such
as [`predict`](@ref) or [`transform`](@ref), can be dispatched.
[`LearnAPI.functions(learner)`](@ref) returns a list of methods that can be applied to
either `learner` or `model`.

For example, a supervised classifier might have a workflow like this:

```julia
model = fit(learner, (X, y))
ŷ = predict(model, Xnew)
```

The signature `fit(learner; verbosity=...)` (no `data`) is provided by learners that do
not generalize to new observations (called *static algorithms*). In that case,
`transform(model, data)` or `predict(model, ..., data)` carries out the actual algorithm
execution, writing any byproducts of that operation to the mutable object `model` returned
by `fit`. Inspect the value of [`LearnAPI.is_static(learner)`](@ref) to determine whether
`fit` consumes `data` or not.

Use `verbosity=0` for warnings only, and `-1` for silent training.

See also [`predict`](@ref), [`transform`](@ref),
[`inverse_transform`](@ref), [`LearnAPI.functions`](@ref), [`obs`](@ref).

# Extended help

# New implementations

Implementation of exactly one of the signatures is compulsory. If `fit(learner;
verbosity=...)` is implemented, then the trait [`LearnAPI.is_static`](@ref) must be
overloaded to return `true`.

The signature must include `verbosity` with `1` as default.

The LearnAPI.jl specification has nothing to say regarding `fit` signatures with more than
two arguments. For convenience, for example, an implementation is free to implement a
slurping signature, such as `fit(learner, X, y, extras...) = fit(learner, (X, y,
extras...))` but LearnAPI.jl does not guarantee such signatures are actually implemented.

## The `target`, `features` and `sees_features` methods

If `data` encapsulates a *target* variable, as defined in LearnAPI.jl documentation, then
[`LearnAPI.target`](@ref) must be implemented. If [`predict`](@ref) or [`transform`](@ref)
are implemented and consume data, then you may need to overload
[`LearnAPI.features`](@ref). If [`predict`](@ref) or [`transform`](@ref) are implemented
and consume no data, then you must instead overload
[`LearnAPI.sees_features(learner)`](@ref) to return `false`, and overload
[`LearnAPI.features(learner, data)`](@ref) to return `nothing`.

$(DOC_DATA_INTERFACE(:fit))

"""
function fit end


# # UPDATE AND COUSINS

"""
    update(model, data, param_replacements...; verbosity=1)

Return an updated version of the `model` object returned by a previous [`fit`](@ref) or
`update` call, but with the specified hyperparameter replacements, in the form `:p1 =>
value1, :p2 => value2, ...`.

```julia
learner = MyForest(ntrees=100)

# train with 100 trees:
model = fit(learner, data)

# add 50 more trees:
model = update(model, data, :ntrees => 150)
```

Provided that `data` is identical with the data presented in a preceding `fit` call *and*
there is at most one hyperparameter replacement, as in the above example, execution is
semantically equivalent to the call `fit(learner, data)`, where `learner` is
`LearnAPI.learner(model)` with the specified replacements. In some cases (typically,
when changing an iteration parameter) there may be a performance benefit to using `update`
instead of retraining ab initio.

If `data` differs from that in the preceding `fit` or `update` call, or there is more than
one hyperparameter replacement, then behaviour is learner-specific.

See also [`fit`](@ref), [`update_observations`](@ref), [`update_features`](@ref).

# New implementations

Implementation is optional. The signature must include `verbosity`. It should be true that
`LearnAPI.learner(newmodel) == newlearner`, where `newmodel` is the return value and
`newlearner = LearnAPI.clone(learner, replacements...)`.


$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.update)"))

See also [`LearnAPI.clone`](@ref)

"""
function update end

"""
    update_observations(model, new_data, param_replacements...; verbosity=1)

Return an updated version of the `model` object returned by a previous [`fit`](@ref) or
`update` call given the new observations present in `new_data`. One may additionally
specify hyperparameter replacements in the form `:p1 => value1, :p2 => value2, ...`.

```julia-repl
learner = MyNeuralNetwork(epochs=10, learning_rate => 0.01)

# train for ten epochs:
model = fit(learner, data)

# train for two more epochs using new data and new learning rate:
model = update_observations(model, new_data, epochs => 12, learning_rate => 0.1)
```

When following the call `fit(learner, data)`, the `update` call is semantically
equivalent to retraining ab initio using a concatenation of `data` and `new_data`,
*provided there are no hyperparameter replacements* (which rules out the example
above). Behaviour is otherwise learner-specific.

See also [`fit`](@ref), [`update`](@ref), [`update_features`](@ref).

# Extended help

# New implementations

Implementation is optional. The signature must include `verbosity`. It should be true that
`LearnAPI.learner(newmodel) == newlearner`, where `newmodel` is the return value and
`newlearner = LearnAPI.clone(learner, replacements...)`.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.update_observations)"))

See also [`LearnAPI.clone`](@ref).

"""
function update_observations end

"""
    update_features(model, new_data, param_replacements...; verbosity=1)
    )

Return an updated version of the `model` object returned by a previous [`fit`](@ref) or
`update` call given the new features encapsulated in `new_data`. One may additionally
specify hyperparameter replacements in the form `:p1 => value1, :p2 => value2, ...`.

When following the call `fit(learner, data)`, the `update` call is semantically
equivalent to retraining ab initio using a concatenation of `data` and `new_data`,
*provided there are no hyperparameter replacements.* Behaviour is otherwise
learner-specific.

See also [`fit`](@ref), [`update`](@ref), [`update_features`](@ref).

# Extended help

# New implementations

Implementation is optional. The signature must include `verbosity`. It should be true that
`LearnAPI.learner(newmodel) == newlearner`, where `newmodel` is the return value and
`newlearner = LearnAPI.clone(learner, replacements...)`.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.update_features)"))

See also [`LearnAPI.clone`](@ref).

"""
function update_features end
