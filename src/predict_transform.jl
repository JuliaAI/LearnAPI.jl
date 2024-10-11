function DOC_IMPLEMENTED_METHODS(name; overloaded=false)
    word = overloaded ? "overloaded" : "implemented"
    "If $word, you must include `$name` in the tuple returned by the "*
    "[`LearnAPI.functions`](@ref) trait. "
end

const OPERATIONS = (:predict, :transform, :inverse_transform)
const DOC_OPERATIONS_LIST_SYMBOL = join(map(op -> "`:$op`", OPERATIONS), ", ")
const DOC_OPERATIONS_LIST_FUNCTION = join(map(op -> "`LearnAPI.$op`", OPERATIONS), ", ")

DOC_MUTATION(op) =
    """

    If [`LearnAPI.is_static(algorithm)`](@ref) is `true`, then `$op` may mutate it's first
    argument, but not in a way that alters the result of a subsequent call to `predict`,
    `transform` or `inverse_transform`. See more at [`fit`](@ref).

    """


DOC_MINIMIZE(func) =
    """

    If, additionally, [`LearnAPI.strip(model)`](@ref) is overloaded, then the following identity
    must hold:

    ```julia
    $func(LearnAPI.strip(model), args...) = $func(model, args...)
    ```

    """

DOC_DATA_INTERFACE(method) =
    """

    ## Assumptions about data

    By default, it is assumed that `data` supports the [`LearnAPI.RandomAccess`](@ref)
    interface; this includes all matrices, with observations-as-columns, most tables, and
    tuples thereof). See [`LearnAPI.RandomAccess`](@ref) for details. If this is not the
    case then an implementation must either: (i) overload [`obs`](@ref) to articulate how
    provided data can be transformed into a form that does support
    [`LearnAPI.RandomAccess`](@ref); or (ii) overload the trait
    [`LearnAPI.data_interface`](@ref) to specify a more relaxed data API. Refer to
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
options with [`LearnAPI.kinds_of_proxy(algorithm)`](@ref), where `algorithm =
LearnAPI.algorithm(model)`.

The shortcut `predict(model, data)` calls the first method with an algorithm-specific
`kind_of_proxy`, namely the first element of [`LearnAPI.kinds_of_proxy(algorithm)`](@ref),
which lists all supported target proxies.

The argument `model` is anything returned by a call of the form `fit(algorithm, ...)`.

If `LearnAPI.features(LearnAPI.algorithm(model)) == nothing`, then argument `data` is
omitted. An example is density estimators.

# Example

In the following, `algorithm` is some supervised learning algorithm with
training features `X`, training target `y`, and test features `Xnew`:

```julia
model = fit(algorithm, (X, y)) # or `fit(algorithm, X, y)`
predict(model, Point(), Xnew)
```

See also [`fit`](@ref), [`transform`](@ref), [`inverse_transform`](@ref).

# Extended help

If `predict` supports data in the form of a tuple `data = (X1, ..., Xn)`, then a slurping
signature is also provided, as in `predict(model, X1, ..., Xn)`.

Note `predict ` does not mutate any argument, except in the special case
`LearnAPI.is_static(algorithm) == true`.

# New implementations

If there is no notion of a "target" variable in the LearnAPI.jl sense, or you need an
operation with an inverse, implement [`transform`](@ref) instead.

Implementation is optional. Only the first signature is implemented, but each
`kind_of_proxy` that gets an implementation must be added to the list returned by
[`LearnAPI.kinds_of_proxy`](@ref).

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.predict)"))

$(DOC_MINIMIZE(:predict))

$(DOC_MUTATION(:predict))

$(DOC_DATA_INTERFACE(:predict))

"""
predict(model, data) = predict(model, kinds_of_proxy(algorithm(model)) |> first, data)
predict(model) = predict(model, kinds_of_proxy(algorithm(model)) |> first)

# automatic slurping of multiple data arguments:
predict(model, k::KindOfProxy, data1, data2, datas...; kwargs...) =
    predict(model, k, (data1, data2, datas...); kwargs...)
predict(model, data1, data2, datas...; kwargs...) =
    predict(model, (data1, data2, datas...); kwargs...)



"""
    transform(model, data)

Return a transformation of some `data`, using some `model`, as returned by
[`fit`](@ref).

For `data` that consists of a tuple, a slurping version is also provided, i.e., you can do
`transform(model, X1, X2, X3)` in place of `transform(model, (X1, X2, X3))`.

# Example

Below, `X` and `Xnew` are data of the same form.

For an `algorithm` that generalizes to new data ("learns"):

```julia
model = fit(algorithm, X; verbosity=0)
transform(model, Xnew)
```

For a static (non-generalizing) transformer:

```julia
model = fit(algorithm)
W = transform(model, X)
```

or, in one step (where supported):

```julia
W = transform(algorithm, X)
```

Note `transform` does not mutate any argument, except in the special case
`LearnAPI.is_static(algorithm) == true`.

See also [`fit`](@ref), [`predict`](@ref),
[`inverse_transform`](@ref).

# Extended help

# New implementations

Implementation for new LearnAPI.jl algorithms is optional. A fallback provides the
slurping version. $(DOC_IMPLEMENTED_METHODS(":(LearnAPI.transform)"))

$(DOC_MINIMIZE(:transform))

$(DOC_MUTATION(:transform))

$(DOC_DATA_INTERFACE(:transform))

"""
transform(model, data1, data2...; kwargs...) =
    transform(model, (data1, datas...); kwargs...) # automatic slurping

"""
    inverse_transform(model, data)

Inverse transform `data` according to some `model` returned by [`fit`](@ref). Here
"inverse" is to be understood broadly, e.g, an approximate
right inverse for [`transform`](@ref).

# Example

In the following, `algorithm` is some dimension-reducing algorithm that generalizes to new
data (such as PCA); `Xtrain` is the training input and `Xnew` the input to be reduced:

```julia
model = fit(algorithm, Xtrain)
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
