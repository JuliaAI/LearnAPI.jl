function DOC_IMPLEMENTED_METHODS(name; overloaded=false)
    word = overloaded ? "overloaded" : "implemented"
    "If $word, you must include `$name` in the tuple returned by the "*
    "[`LearnAPI.functions`](@ref) trait. "
end

DOC_MUTATION(op) =
    """

    If [`LearnAPI.is_static(learner)`](@ref) is `true`, then `$op` may mutate it's first
    argument (to record byproducts of the computation not naturally part of the return
    value) but not in a way that alters the result of a subsequent call to `predict`,
    `transform` or `inverse_transform`. See more at [`fit`](@ref).

    """

DOC_SLURPING(op) =
    """

    An implementation is free to implement `$op` signatures with additional positional
 arguments (eg., data-slurping signatures) but LearnAPI.jl is silent about their
 interpretation or existence.

    """

DOC_MINIMIZE(func) =
    """

    If, additionally, [`LearnAPI.strip(model)`](@ref) is overloaded, then the following
    identity must hold:

    ```julia
    $func(LearnAPI.strip(model), args...) == $func(model, args...)
    ```

    """

DOC_DATA_INTERFACE(method) =
    """

    ## Assumptions about data

    By default, it is assumed that `data` supports the [`LearnAPI.RandomAccess`](@ref)
    interface; this includes all matrices, with observations-as-columns, most tables, and
    tuples thereof. See [`LearnAPI.RandomAccess`](@ref) for details. If this is not the
    case then an implementation must either: (i) overload [`obs`](@ref) to articulate how
    provided data can be transformed into a form that does support
    [`LearnAPI.RandomAccess`](@ref); or (ii) overload the trait
    [`LearnAPI.data_interface`](@ref) to specify a more relaxed data API. Refer tbo
    document strings for details.

    """


# # METHOD STUBS/FALLBACKS

"""
    predict(model, kind_of_proxy::LearnAPI.KindOfProxy, data)
    predict(model, data)

The first signature returns target predictions, or proxies for target predictions, for
input features `data`, according to some `model` returned by [`fit`](@ref). Where
supported, these are literally target predictions if `kind_of_proxy = Point()`,
and probability density/mass functions if `kind_of_proxy = Distribution()`. List all
options with [`LearnAPI.kinds_of_proxy(learner)`](@ref), where `learner =
LearnAPI.learner(model)`.

```julia
model = fit(learner, (X, y))
predict(model, Point(), Xnew)
```

The shortcut `predict(model, data)` calls the first method with learner-specific
`kind_of_proxy`, namely the first element of [`LearnAPI.kinds_of_proxy(learner)`](@ref),
which lists all supported target proxies.

The argument `model` is anything returned by a call of the form `fit(learner, ...)`.

If `LearnAPI.features(LearnAPI.learner(model)) == nothing`, then the argument `data` is
omitted in both signatures. An example is density estimators.

See also [`fit`](@ref), [`transform`](@ref), [`inverse_transform`](@ref).

# Extended help

In the special case `LearnAPI.is_static(learner) == true`, it is possible that
`predict(model, ...)` will mutate `model`, but not in a way that affects subsequent
`predict` calls.

# New implementations

If there is no notion of a "target" variable in the LearnAPI.jl sense, or you need an
operation with an inverse, implement [`transform`](@ref) instead.

Implementation is optional. Only the first signature (with or without the `data` argument)
is implemented, but each `kind_of_proxy::`[`KindOfProxy`](@ref) that gets an
implementation must be added to the list returned by
[`LearnAPI.kinds_of_proxy(learner)`](@ref). List all available kinds of proxy by doing
`LearnAPI.kinds_of_proxy()`.

If `data` is not present in the implemented signature (eg., for density estimators) then
[`LearnAPI.features(learner, data)`](@ref) must return `nothing`.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.predict)"))

$(DOC_MINIMIZE(:predict))

$(DOC_MUTATION(:predict))

$(DOC_DATA_INTERFACE(:predict))

"""
predict(model, data) = predict(model, kinds_of_proxy(learner(model)) |> first, data)
predict(model) = predict(model, kinds_of_proxy(learner(model)) |> first)

"""
    transform(model, data)

Return a transformation of some `data`, using some `model`, as returned by
[`fit`](@ref).

# Example

Below, `X` and `Xnew` are data of the same form.

For a `learner` that generalizes to new data ("learns"):

```julia
model = fit(learner, X; verbosity=0)
transform(model, Xnew)
```

or, in one step (where supported):

```julia
W = transform(learner, X) # `fit` implied
```

For a static (non-generalizing) transformer:

```julia
model = fit(learner)
W = transform(model, X)
```

or, in one step (where supported):

```julia
W = transform(learner, X) # `fit` implied
```

In the special case `LearnAPI.is_static(learner) == true`, it is possible that
`transform(model, ...)` will mutate `model`, but not in a way that affects subsequent
`transform` calls.

See also [`fit`](@ref), [`predict`](@ref),
[`inverse_transform`](@ref).

# Extended help

# New implementations

Implementation for new LearnAPI.jl learners is
optional. $(DOC_IMPLEMENTED_METHODS(":(LearnAPI.transform)"))

$(DOC_SLURPING(:transform))

$(DOC_MINIMIZE(:transform))

$(DOC_MUTATION(:transform))

$(DOC_DATA_INTERFACE(:transform))

"""
function transform end


"""
    inverse_transform(model, data)

Inverse transform `data` according to some `model` returned by [`fit`](@ref). Here
"inverse" is to be understood broadly, e.g, an approximate
right or left inverse for [`transform`](@ref).

# Example

In the following, `learner` is some dimension-reducing algorithm that generalizes to new
data (such as PCA); `Xtrain` is the training input and `Xnew` the input to be reduced:

```julia
model = fit(learner, Xtrain)
W = transform(model, Xnew)       # reduced version of `Xnew`
Ŵ = inverse_transform(model, W)  # embedding of `W` in original space
```

See also [`fit`](@ref), [`transform`](@ref), [`predict`](@ref).

# Extended help

# New implementations

Implementation is optional. $(DOC_IMPLEMENTED_METHODS(":(LearnAPI.inverse_transform)"))

$(DOC_MINIMIZE(:inverse_transform))

"""
function inverse_transform end
