"""
    LearnAPI.input(algorithm, data)

Where `data` is a supported data argument for `fit`, extract from `data` something
suitable for passing as the third argument to `predict`, as in the following sample
workflow:

```julia
model = fit(algorithm, data)
X = input(data)
ŷ = predict(algorithm, kind_of_proxy, X) # eg, `kind_of_proxy = LiteralTarget()`
```

The return value has the same number of observations as `data` does. Where
`LearnAPI.target(algorithm)` is `true` (supervised learning) one expects `ŷ` above to be
an approximate proxy for `target(algorithm, data)`, the training target.


# New implementations

The following fallbacks typically make overloading `LearnAPI.input` unnecessary:

```julia
LearnAPI.input(algorithm, data) = data
LearnAPI.input(algorithm, data::Tuple) = first(data)
```

"""
input(algorithm, data) = data
input(algorithm, data::Tuple) = first(data)
