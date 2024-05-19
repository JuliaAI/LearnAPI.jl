# [`predict`, `transform` and `inverse_transform`](@id operations)

```julia
predict(model, kind_of_proxy, data)
transform(model, data)
inverse_transform(model, data)
```

When a method expects a tuple form of argument, `data = (X1, ..., Xn)`, then a slurping
signature is also provided, as in `transform(model, X1, ..., Xn)`.

## Typical worklows

Train some supervised `algorithm`:

```julia
model = fit(algorithm, X, y)
```

Predict probability distributions:

```julia
ŷ = predict(model, Distribution(), Xnew)
```

Generate point predictions:

```julia
ŷ = predict(model, LiteralTarget(), Xnew)
```

Train a dimension-reducing `algorithm`:

```julia
model = fit(algorithm, X)
Xnew_reduced = transform(model, Xnew)
```

Apply an approximate right inverse:

```julia
inverse_transform(model, Xnew_reduced)
```

### An advanced workflow

```julia
fitobs = obs(algorithm, (X, y)) # algorithm-specific repr. of data
model = fit(algorithm, MLUtils.getobs(fitobs, 1:100))
predictobs = obs(model, MLUtils.getobs(X, 101:150))
ŷ = predict(model, LiteralTarget(), predictobs)
```


## Implementation guide

| method                      | compulsory? | fallback |
|:----------------------------|:-----------:|:--------:|
| [`predict`](@ref)           | no          | none     |
| [`transform`](@ref)         | no          | none     |
| [`inverse_transform`](@ref) | no          | none     |

### Predict or transform?

If the algorithm has a notion of [target variable](@ref proxy), then use 
[`predict`](@ref) to output each supported [kind of target proxy](@ref
proxy_types) (`LiteralTarget()`, `Distribution()`, etc).

For output not associated with a target variable, implement [`transform`](@ref)
instead, which does not dispatch on [`LearnAPI.KindOfProxy`](@ref), but can be optionally
paired with an implementation of [`inverse_transform`](@ref), for returning (approximate)
right inverses to `transform`.


## Reference

```@docs
predict
transform
inverse_transform
```