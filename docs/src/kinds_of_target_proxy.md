# [Kinds of Target Proxy](@id proxy_types)

The available kinds of [target proxy](@ref proxy) (used for `predict` dispatch) are
classified by subtypes of `LearnAPI.KindOfProxy`. These types are intended for dispatch
only and have no fields.

```@docs
LearnAPI.KindOfProxy
```

## Simple target proxies

```@docs
LearnAPI.IID
```

## Proxies for density estimation algorithms

```@docs
LearnAPI.Single
```

## Joint probability distributions

```@docs
LearnAPI.Joint
```
