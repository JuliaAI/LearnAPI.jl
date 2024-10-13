# # FIT

"""
    fit(algorithm, data; verbosity=1)
    fit(algorithm; verbosity=1)

Execute the algorithm with configuration `algorithm` using the provided training `data`,
returning an object, `model`, on which other methods, such as [`predict`](@ref) or
[`transform`](@ref), can be dispatched.  [`LearnAPI.functions(algorithm)`](@ref) returns a
list of methods that can be applied to either `algorithm` or `model`.

The second signature is provided by algorithms that do not generalize to new observations
(called *static algorithms*). In that case, `transform(model, data)` or `predict(model,
..., data)` carries out the actual algorithm execution, writing any byproducts of that
operation to the mutable object `model` returned by `fit`.

For example, a supervised classifier might have a workflow like this:

```julia
model = fit(algorithm, (X, y))
ŷ = predict(model, Xnew)
```

Use `verbosity=0` for warnings only, and `-1` for silent training.

See also [`predict`](@ref), [`transform`](@ref), [`inverse_transform`](@ref),
[`LearnAPI.functions`](@ref), [`obs`](@ref).

# Extended help

# New implementations

Implementation of exactly one of the signatures is compulsory. If `fit(algorithm;
verbosity=1)` is implemented, then the trait [`LearnAPI.is_static`](@ref) must be
overloaded to return `true`.

The signature must include `verbosity`.

If `data` encapsulates a *target* variable, as defined in LearnAPI.jl documentation, then
[`LearnAPI.target(data)`] must be overloaded to return it. If [`predict`](@ref) or
[`transform`](@ref) are implemented and consume data, then
[`LearnAPI.features(data)`](@ref) must return something that can be passed as data to
these methods. A fallback returns `first(data)` if `data` is a tuple, and `data`
otherwise`.

The LearnAPI.jl specification has nothing to say regarding `fit` signatures with more than
two arguments. For convenience, for example, an algorithm is free to implement a slurping
signature, such as `fit(algorithm, X, y, extras...) = fit(algorithm, (X, y, extras...))` but
LearnAPI.jl does not guarantee such signatures are actually implemented.

$(DOC_DATA_INTERFACE(:fit))

"""
function fit end


# # UPDATE AND COUSINS

"""
    update(model, data; verbosity=1, hyperparam_replacements...)

Return an updated version of the `model` object returned by a previous [`fit`](@ref) or
`update` call, but with the specified hyperparameter replacements, in the form `p1=value1,
p2=value2, ...`.

Provided that `data` is identical with the data presented in a preceding `fit` call *and*
there is at most one hyperparameter replacement, as in the example below, execution is
semantically equivalent to the call `fit(algorithm, data)`, where `algorithm` is
`LearnAPI.algorithm(model)` with the specified replacements. In some cases (typically,
when changing an iteration parameter) there may be a performance benefit to using `update`
instead of retraining ab initio.

If `data` differs from that in the preceding `fit` or `update` call, or there is more than
one hyperparameter replacement, then behaviour is algorithm-specific.

```julia
algorithm = MyForest(ntrees=100)

# train with 100 trees:
model = fit(algorithm, data)

# add 50 more trees:
model = update(model, data; ntrees=150)
```

See also [`fit`](@ref), [`update_observations`](@ref), [`update_features`](@ref).

# New implementations

Implementation is optional. The signature must include
`verbosity`. $(DOC_IMPLEMENTED_METHODS(":(LearnAPI.update)"))

See also [`LearnAPI.clone`](@ref)

"""
function update end

"""
    update_observations(model, new_data; verbosity=1, parameter_replacements...)

Return an updated version of the `model` object returned by a previous [`fit`](@ref) or
`update` call given the new observations present in `new_data`. One may additionally
specify hyperparameter replacements in the form `p1=value1, p2=value2, ...`.

When following the call `fit(algorithm, data)`, the `update` call is semantically
equivalent to retraining ab initio using a concatenation of `data` and `new_data`,
*provided there are no hyperparameter replacements.* Behaviour is otherwise
algorithm-specific.

```julia-repl
algorithm = MyNeuralNetwork(epochs=10, learning_rate=0.01)

# train for ten epochs:
model = fit(algorithm, data)

# train for two more epochs using new data and new learning rate:
model = update_observations(model, new_data; epochs=2, learning_rate=0.1)
```

See also [`fit`](@ref), [`update`](@ref), [`update_features`](@ref).

# Extended help

# New implementations

Implementation is optional. The signature must include
`verbosity`. $(DOC_IMPLEMENTED_METHODS(":(LearnAPI.update_observations)"))

See also [`LearnAPI.clone`](@ref).

"""
function update_observations end

"""
    update_features(model, new_data; verbosity=1, parameter_replacements...)

Return an updated version of the `model` object returned by a previous [`fit`](@ref) or
`update` call given the new features encapsulated in `new_data`. One may additionally
specify hyperparameter replacements in the form `p1=value1, p2=value2, ...`.

When following the call `fit(algorithm, data)`, the `update` call is semantically
equivalent to retraining ab initio using a concatenation of `data` and `new_data`,
*provided there are no hyperparameter replacements.* Behaviour is otherwise
algorithm-specific.

See also [`fit`](@ref), [`update`](@ref), [`update_features`](@ref).

# Extended help

# New implementations

Implementation is optional. The signature must include
`verbosity`. $(DOC_IMPLEMENTED_METHODS(":(LearnAPI.update_features)"))

See also [`LearnAPI.clone`](@ref).

"""
function update_features end
