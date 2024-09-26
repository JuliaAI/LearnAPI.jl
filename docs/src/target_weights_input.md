# [`input`](@id input)

```julia
LearnAPI.input(algorithm, data) -> <suitable input for `predict`>
```

# Typical workflow

Not typically appearing in a general user's workflow but useful in meta-alagorithms, such
as cross-validation (see the example in [`obs` and Data Interfaces](@ref data_interface)).

Supposing `algorithm` is a supervised classifier predicting a one-dimensional vector
target:

```julia
model = fit(algorithm, data)
X = LearnAPI.input(algorithm, data)
y = LearnAPI.target(algorithm, data)
ŷ = predict(model, LiteralTarget(), X)
training_loss = sum(ŷ .!= y)
```

# Implementation guide

The fallback returns `first(data)`, assuming `data` is a tuple, and `data` otherwise.

| method                   | compulsory? |
|:-------------------------|:-----------:|
| [`LearnAPI.input`](@ref) | no          |

# Reference

```@docs
LearnAPI.input
```
