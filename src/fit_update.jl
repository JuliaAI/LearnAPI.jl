# # FIT

"""
    fit(learner, data; verbosity=LearnAPI.default_verbosity())
    fit(learner; verbosity=LearnAPI.default_verbosity()))

In the case of the first signature, execute the machine learning or statistical algorithm
with configuration `learner` using the provided training `data`, returning an object,
`model`, on which other methods, such as [`predict`](@ref) or [`transform`](@ref), can be
dispatched.  [`LearnAPI.functions(learner)`](@ref) returns a list of methods that can be
applied to either `learner` or `model`.

For example, a supervised classifier might have a workflow like this:

```julia
model = fit(learner, (X, y))
ŷ = predict(model, Xnew)
```

Use `verbosity=0` for warnings only, and `-1` for silent training.

This `fit` signature applies to all learners for which [`LearnAPI.kind_of(learner)`](@ref)
returns [`LearnAPI.Descriminative()`](@ref) or [`LearnAPI.Generative()`](@ref).

# Static learners

In the case of a learner that does not generalize to new data, the second `fit` signature
can be used to wrap the `learner` in an object called `model` that the calls
`transform(model, data)` or `predict(model, ..., data)` may mutate, so as to record
byproducts of the core algorithm specified by `learner`, before returning the outcomes of
primary interest.

Here's a sample workflow:

```julia
model = fit(learner)       # e.g, `learner` specifies DBSCAN clustering parameters
labels = predict(model, X) # compute and return cluster labels for `X`
LearnAPI.extras(model)     # return outliers in the data `X`
```
This `fit` signature applies to all learners for which
[`LearnAPI.kind_of(learner)`](@ref)` == `[`LearnAPI.Static()`](@ref).

See also [`predict`](@ref), [`transform`](@ref), [`inverse_transform`](@ref),
[`LearnAPI.functions`](@ref), [`obs`](@ref), [`LearnAPI.kind_of`](@ref).

# Extended help

# New implementations

Implementation of exactly one of the signatures is compulsory. Unless implementing the
[`LearnAPI.Descriminative()`](@ref) `fit`/`predict`/`transform` pattern,
[`LearnAPI.kind_of(learner)`](@ref) will need to be suitably overloaded.

The `fit` signature must include the keyword argument `verbosity` with
`LearnAPI.default_verbosity()` as default.

The LearnAPI.jl specification has nothing to say regarding `fit` signatures with more than
two arguments. For convenience, for example, an implementation is free to implement a
slurping signature, such as `fit(learner, X, y, extras...) = fit(learner, (X, y,
extras...))` but LearnAPI.jl does not guarantee such signatures are actually implemented.

## The `target` and `features` methods

If [`LearnAPI.kind_of(learner)`](@ref) returns [`LearnAPI.Descriminative()`](@ref) or
[`LearnAPI.Generative()`](@ref) then the methods [`LearnAPI.target`](@ref) and/or
[`LearnAPI.features`](@ref), which deconstruct the form of `data` consumed by `fit`, may
require overloading. Refer to their document strings for details.

$(DOC_DATA_INTERFACE(:fit))

"""
function fit end


# # UPDATE AND COUSINS

"""
    update(model, data, param_replacements...; verbosity=LearnAPI.default_verbosity())

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

Implementation is optional. The signature must include the `verbosity` keyword
argument. It should be true that `LearnAPI.learner(newmodel) == newlearner`, where
`newmodel` is the return value and `newlearner = LearnAPI.clone(learner,
replacements...)`.

Cannot be implemented if [`LearnAPI.kind_of(learner)`](@ref)` == `LearnAPI.Static()`.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.update)"))

See also [`LearnAPI.clone`](@ref)

"""
function update end

"""
    update_observations(
       model,
       new_data,
       param_replacements...;
       verbosity=LearnAPI.default_verbosity(),
    )

Return an updated version of the `model` object returned by a previous [`fit`](@ref) or
`update` call, given the new observations present in `new_data`. One may additionally
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

Implementation is optional. The signature must include the `verbosity` keyword
argument. It should be true that `LearnAPI.learner(newmodel) == newlearner`, where
`newmodel` is the return value and `newlearner = LearnAPI.clone(learner,
replacements...)`.

Cannot be implemented if [`LearnAPI.kind_of(learner)`](@ref)` == `LearnAPI.Static()`.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.update_observations)"))

See also [`LearnAPI.clone`](@ref).

"""
function update_observations end

"""
    update_features(
        model,
        new_data,
        param_replacements,...;
        verbosity=LearnAPI.default_verbosity(),
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

Implementation is optional. The signature must include the `verbosity` keyword
argument. It should be true that `LearnAPI.learner(newmodel) == newlearner`, where
`newmodel` is the return value and `newlearner = LearnAPI.clone(learner,
replacements...)`.

Cannot be implemented if [`LearnAPI.kind_of(learner)`](@ref)` == `LearnAPI.Static()`.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.update_features)"))

See also [`LearnAPI.clone`](@ref).

"""
function update_features end
