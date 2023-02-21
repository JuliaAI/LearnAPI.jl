# [Reference](@id reference)

> **Summary** In LearnAPI.jl an **algorithm** is a container for hyperparameters of some
> ML/Statistics algorithm (which may or may not "learn"). Functionality is created by
> overloading **methods** provided by the interface, which are divided into training
> methods (e.g., `fit`), operations (e.g.,. `predict` and `transform`) and accessor
> functions (e.g., `feature_importances`). Promises of particular behavior are articulated
> by **algorithm traits**.

Here we give the definitive specification of the interface provided by LearnAPI.jl. For a
more informal guide see  [Anatomy of an Implementation](@ref) and [Common Implementation Patterns](@ref).

!!! important

    The reader is assumed to be familiar with the LearnAPI-specific meanings of     the following terms, as outlined in 
	[Scope and undefined notions](@ref scope): **data**, **metadata**, 
	**hyperparameter**, **observation**, **target**, **target proxy**.
	
## Algorithms

In LearnAPI.jl an **algorithm** is some julia object `alg` storing the hyperparameters of
some algorithm from machine learning or statistics used manipulating data. Typically the
algorithm "learns" from data in a training event, but this is not essential; "static" data
processing, with parameters, is included. 

The type of `alg` will have a name reflecting that of the algorithm, such as
`DecisionTreeRegressor`.

Additionally, for `alg::Alg` to be a LearnAPI algorithm, we require:

- `Base.propertynames(alg)` returns the hyperparameters of `alg`.

- If `alg` is an algorithm, then so are all instances of the same type.

- If `_alg` is another algorithm, then `alg == _alg` if and only if `typeof(alg) == typeof(_alg)` and
  corresponding properties are `==`. This includes properties that are random number
  generators (which should be copied in training to avoid mutation).

- If an algorithm has other algorithms as hyperparameters, then [`LearnAPI.is_wrapper`](@ref)`(alg)`
  must be `true`.

- A keyword constructor for `Alg` exists, providing default values for *all* non-algorithm
  hyperparameters.

Whenever any LearnAPI method (excluding traits) is overloaded for some type `Alg` (e.g.,
`predict`, `transform`, `fit`) then that is a promise that all instances of `Alg` are
algorithms (and the trait [`LearnAPI.functions`](@ref)`(Alg)` will be non-empty).

It is supposed that making copies of algorithm objects is a cheap operation. Consequently,
*learned* parameters, such as weights in a neural network (the `fitted_params` described
in [Fit, update! and ingest!](@ref)) should not be stored in the algorithm object. Storing
learned parameters in an algorithm is not explicitly ruled out, but doing so might lead to
performance issues in packages adopting LearnAPI.jl.


### Example

Any instance of `GradientRidgeRegressor` defined below is a valid LearnAPI.jl algorithm:

```julia
struct GradientRidgeRegressor{T<:Real} <: LearnAPI.Algorithm
    learning_rate::T
    epochs::Int
    l2_regularization::T
end
```

The same is true if we omit the subtyping `<: LearnAPI.Algorithm`, but not if we also make
this a `mutable struct`. In that case we will need to overload `Base.==` for
`GradientRidgeRegressor`.

```@docs
LearnAPI.Algorithm
```

## Methods

None of the methods described in the linked sections below are compulsory, but any
implemented or overloaded method that is not an algorithm trait must be added to the return
value of [`LearnAPI.functions`](@ref), as in

```julia
LearnAPI.functions(::Type{<SomeAlgorithmType}) = (:fit, :update!, :predict)
```

or using the shorthand

```julia
@trait SomeAlgorithmType functions=(:fit, :update!, :predict)
```

- [Fit, update! and ingest!](@ref) (training methods): for algorithms that "learn" (generalize
  to new data)

- [Operations](@ref operations): `predict`, `transform` and their relatives

- [Accessor Functions](@ref): accessing certain byproducts of training that many algorithms
  share, such as feature importances and training losses
  
- [Optional Data Interface](@ref data_interface): `getobs` and `reformat`

- [Algorithm Traits](@ref): contracts for specific behavior, such as "The second data
  argument of `fit` is a target variable" or "I predict probability distributions"
  
