 function DOC_IMPLEMENTED_METHODS(name; overloaded=false)
    word = overloaded ? "overloaded" : "implemented"
    "If $word, you must include `$name` in the tuple returned by the "*
    "[`LearnAPI.functions`](@ref) trait. "
end

const OPERATIONS = (:predict, :transform, :inverse_transform)
const DOC_OPERATIONS_LIST_SYMBOL = join(map(op -> "`:$op`", OPERATIONS), ", ")
const DOC_OPERATIONS_LIST_FUNCTION = join(map(op -> "`LearnAPI.$op`", OPERATIONS), ", ")

DOC_ARGUMENTS(func) =
"""
- `data`: tuple of data objects with a common number of observations, for example,
  `data = (X, y, w)` where `X` is a table of features, `y` is a target vector with the
  same number of rows, and `w` a vector of per-observation weights.

"""

DOC_MUTATION(op) =
    """

    If [`LearnAPI.predict_or_transform_mutates(algorithm)`](@ref) is overloaded to return
    `true`, then `$op` may mutate it's first argument, but not in a way that alters the
    result of a subsequent call to `obspredict`, `obstransform` or
    `inverse_transform`. This is necessary for some non-generalizing algorithms but is
    otherwise discouraged. See more at [`fit`](@ref).

    """


DOC_MINIMIZE(func) =
    """

    If, additionally, [`minimize(model)`](@ref) is overloaded, then the following identity
    must hold:

    ```julia
    $func(minimize(model), args...) = $func(model, args...)
    ```

    """

# # METHOD STUBS/FALLBACKS

"""
    predict(model, kind_of_proxy::LearnAPI.KindOfProxy, data...)
    predict(model, data...)

The first signature returns target or target proxy predictions for input features `data`,
according to some `model` returned by [`fit`](@ref) or [`obsfit`](@ref). Where supported,
these are literally target predictions if `kind_of_proxy = LiteralTarget()`, and
probability density/mass functions if `kind_of_proxy = Distribution()`. List all options
with [`LearnAPI.kinds_of_proxy(algorithm)`](@ref), where `algorithm =
LearnAPI.algorithm(model)`.

The shortcut `predict(model, data...) = predict(model, LiteralTarget(), data...)` is also
provided.

# Arguments

- `model` is anything returned by a call of the form `fit(algorithm, ...)`, for some
  LearnAPI-complaint `algorithm`.

$(DOC_ARGUMENTS(:predict))

# Example

In the following, `algorithm` is some supervised learning algorithm with
training features `X`, training target `y`, and test features `Xnew`:

```julia
model = fit(algorithm, X, y; verbosity=0)
predict(model, LiteralTarget(), Xnew)
```

Note `predict ` does not mutate any argument, except in the special case
`LearnAPI.predict_or_transform_mutates(algorithm) = true`.

See also [`obspredict`](@ref), [`fit`](@ref), [`transform`](@ref),
[`inverse_transform`](@ref).

# Extended help

# New implementations

LearnAPI.jl provides the following definition of `predict` which is never to be directly
overloaded:

```julia
predict(model, kop::LearnAPI.KindOfProxy, data...) =
    obspredict(model, kop, obs(predict, LearnAPI.algorithm(model), data...))
```

Rather, new algorithms overload [`obspredict`](@ref).

"""
predict(model, kind_of_proxy::KindOfProxy, data...) =
    obspredict(model, kind_of_proxy, obs(predict, algorithm(model), data...))
predict(model, data...) = predict(model, LiteralTarget(), data...)

"""
    obspredict(model, kind_of_proxy::LearnAPI.KindOfProxy, obsdata)

Similar to `predict` but consumes algorithm-specific representations of input data,
`obsdata`, as returned by `obs(predict, algorithm, data...)`. Here `data...` is the form of
data expected in the main [`predict`](@ref) method.  Alternatively, such `obsdata` may be
replaced by a resampled version, where resampling is performed using `MLUtils.getobs`
(always supported).

For some algorithms and workflows, `obspredict` will have a performance benefit over
[`predict`](@ref). See more at [`obs`](@ref).

# Example

In the following, `algorithm` is some supervised learning algorithm with
training features `X`, training target `y`, and test features `Xnew`:

```julia
model = fit(algorithm, X, y)
obsdata = obs(predict, algorithm, Xnew)
ŷ = obspredict(model, LiteralTarget(), obsdata)
@assert ŷ == predict(model, LiteralTarget(), Xnew)
```

See also [`predict`](@ref), [`fit`](@ref), [`transform`](@ref),
[`inverse_transform`](@ref), [`obs`](@ref).

# Extended help

# New implementations

Implementation of `obspredict` is optional, but required to enable `predict`. The method
must also handle `obsdata` in the case it is replaced by `MLUtils.getobs(obsdata, I)` for
some collection `I` of indices. If [`obs`](@ref) is not overloaded, then `obsdata = data`,
where `data...` is what the standard [`predict`](@ref) call expects, as in the call
`predict(model, kind_of_proxy, data...)`. Note `data` is always a tuple, even if `predict`
has only one data argument. See more at [`obs`](@ref).


$(DOC_MUTATION(:obspredict))

If overloaded, you must include both `LearnAPI.obspredict` and `LearnAPI.predict` in the
list of methods returned by the [`LearnAPI.functions`](@ref) trait.

An implementation is provided for each kind of target proxy you wish to support. See the
LearnAPI.jl documentation for options. Each supported `kind_of_proxy` instance should be
listed in the return value of the [`LearnAPI.kinds_of_proxy(algorithm)`](@ref) trait.

$(DOC_MINIMIZE(:obspredict))

"""
function obspredict end

"""
    transform(model, data...)

Return a transformation of some `data`, using some `model`, as returned by [`fit`](@ref).

# Arguments

- `model` is anything returned by a call of the form `fit(algorithm, ...)`, for some
  LearnAPI-complaint `algorithm`.

$(DOC_ARGUMENTS(:transform))

# Example

Here `X` and `Xnew` are data of the same form:

```julia
# For an algorithm that generalizes to new data ("learns"):
model = fit(algorithm, X; verbosity=0)
transform(model, Xnew)

# For a static (non-generalizing) transformer:
model = fit(algorithm)
transform(model, X)
```

Note `transform` does not mutate any argument, except in the special case
`LearnAPI.predict_or_transform_mutates(algorithm) = true`.

See also [`obstransform`](@ref), [`fit`](@ref), [`predict`](@ref),
[`inverse_transform`](@ref).

# Extended help

# New implementations

LearnAPI.jl provides the following definition of `transform` which is never to be directly
overloaded:


```julia
transform(model, data...) =
    obstransform(model, obs(predict, LearnAPI.algorithm(model), data...))
```

Rather, new algorithms overload [`obstransform`](@ref).

"""
transform(model, data...) =
    obstransform(model, obs(transform, LearnAPI.algorithm(model), data...))

"""
    obstransform(model, kind_of_proxy::LearnAPI.KindOfProxy, obsdata)

Similar to `transform` but consumes algorithm-specific representations of input data,
`obsdata`, as returned by `obs(transform, algorithm, data...)`. Here `data...` is the
form of data expected in the main [`transform`](@ref) method.  Alternatively, such
`obsdata` may be replaced by a resampled version, where resampling is performed using
`MLUtils.getobs` (always supported).

For some algorithms and workflows, `obstransform` will have a performance benefit over
[`transform`](@ref). See more at [`obs`](@ref).

# Example

In the following, `algorithm` is some unsupervised learning algorithm with
training features `X`, and test features `Xnew`:

```julia
model = fit(algorithm, X, y)
obsdata = obs(transform, algorithm, Xnew)
W = obstransform(model, obsdata)
@assert W == transform(model, Xnew)
```

See also [`transform`](@ref), [`fit`](@ref), [`predict`](@ref),
[`inverse_transform`](@ref), [`obs`](@ref).

# Extended help

# New implementations

Implementation of `obstransform` is optional, but required to enable `transform`. The
method must also handle `obsdata` in the case it is replaced by `MLUtils.getobs(obsdata,
I)` for some collection `I` of indices. If [`obs`](@ref) is not overloaded, then `obsdata
= data`, where `data...` is what the standard [`transform`](@ref) call expects, as in the
call `transform(model, data...)`. Note `data` is always a tuple, even if `transform` has
only one data argument. See more at [`obs`](@ref).

$(DOC_MUTATION(:obstransform))

If overloaded, you must include both `LearnAPI.obstransform` and `LearnAPI.transform` in
the list of methods returned by the [`LearnAPI.functions`](@ref) trait.

Each supported `kind_of_proxy` should be listed in the return value of the
[`LearnAPI.kinds_of_proxy(algorithm)`](@ref) trait.

$(DOC_MINIMIZE(:obstransform))
"""
function obstransform end

"""
    inverse_transform(model, data)

Inverse transform `data` according to some `model` returned by [`fit`](@ref). Here
"inverse" is to be understood broadly, e.g, an approximate
right inverse for [`transform`](@ref).

# Arguments

- `model`: anything returned by a call of the form `fit(algorithm, ...)`, for some
  LearnAPI-complaint `algorithm`.

- `data`: something having the same form as the output of `transform(model, inputs...)`

# Example

In the following, `algorithm` is some dimension-reducing algorithm that generalizes to new
data (such as PCA); `Xtrain` is the training input and `Xnew` the input to be reduced:

```julia
model = fit(algorithm, Xtrain; verbosity=0)
W = transform(model, Xnew)       # reduced version of `Xnew`
Ŵ = inverse_transform(model, W)  # embedding of `W` in original space
```

See also [`fit`](@ref), [`transform`](@ref), [`predict`](@ref).

# Extended help

# New implementations

Implementation is optional. $(DOC_IMPLEMENTED_METHODS(:inverse_transform, ))

$(DOC_MINIMIZE(:inverse_transform))

"""
function inverse_transform end
