```@raw html
<script async defer src="https://buttons.github.io/buttons.js"></script>
<span style="color: #9558B2;font-size:4.5em;">
LearnAPI.jl</span>
<br>
<span style="color: #9558B2;font-size:1.6em;font-style:italic;">
A base Julia interface for machine learning and statistics </span>
<br>
<br>
```

LearnAPI.jl is a lightweight, functional-style interface, providing a collection of
[methods](@ref Methods), such as `fit` and `predict`, to be implemented by algorithms from
machine learning and statistics. Through such implementations, these algorithms buy into
functionality, such as hyperparameter optimization, as provided by ML/statistics toolboxes
and other packages. LearnAPI.jl also provides a number of Julia [traits](@ref traits) for
promising specific behavior.

```@raw html
&#128679;
```

!!! warning

    The API described here is under active development and not ready for adoption. 
	Join an ongoing design discussion at 
	[this](https://discourse.julialang.org/t/ann-learnapi-jl-proposal-for-a-basement-level-machine-learning-api/93048) 
	Julia Discourse thread.
	

## Sample workflow

Suppose `forest` is some object encapsulating the hyperparameters of the [random forest
algorithm](https://en.wikipedia.org/wiki/Random_forest) (the number of trees,
etc.). Then, a LearnAPI.jl interface can be implemented, for objects with the type of
`forest`, to enable the following basic workflow:

```julia
X = <some training features>
y = <some training target>
Xnew = <some test or production features>

# Train:
model = fit(forest, X, y)

# Predict probability distributions:
predict(model, Distribution(), Xnew)

# Generate point predictions:
ŷ = predict(model, LiteralTarget(), Xnew) # or `predict(model, Xnew)`

# Apply an "accessor function" to inspect byproducts of training:
LearnAPI.feature_importances(model)

# Slim down and otherwise prepare model for serialization:
small_model = minimize(model)
serialize("my_random_forest.jls", small_model)

# Recover saved model and algorithm configuration:
recovered_model = deserialize("my_random_forest.jls")
@assert LearnAPI.algorithm(recovered_model) == forest
@assert predict(recovered_model, LiteralTarget(), Xnew) == ŷ
```

`Distribution` and `LiteralTarget` are singleton types owned by LearnAPI.jl. They allow
dispatch based on the [kind of target proxy](@ref proxy), a key LearnAPI.jl concept.
LearnAPI.jl places more emphasis on the notion of target variables and target proxies than
on the usual supervised/unsupervised learning dichotomy. From this point of view, a
supervised algorithm is simply one in which a target variable exists, and happens to
appear as an input to training but not to prediction.

In LearnAPI.jl, a method called [`obs`](@ref data_interface) gives users access to an
"internal", algorithm-specific, representation of input data, which is always
"observation-accessible", in the sense that it can be resampled using
[MLUtils.jl](https://github.com/JuliaML/MLUtils.jl) `getobs/numobs` interface. The
implementation can arrange for this resampling to be efficient, and workflows based on
`obs` can have performance benefits.

## Learning more

- [Anatomy of an Implementation](@ref): informal introduction to the main actors in a new
  LearnAPI.jl implementation

- [Reference](@ref reference): official specification

- [Common Implementation Patterns](@ref): implementation suggestions for common,
  informally defined, algorithm types

- [Testing an Implementation](@ref)
