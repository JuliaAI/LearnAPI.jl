```@raw html
<script async defer src="https://buttons.github.io/buttons.js"></script>

<div style="font-size:1.4em;font-weight:bold;">
  <a href="anatomy_of_an_implementation"
    style="color: #389826;">Tutorial</a>           &nbsp;|&nbsp;
  <a href="common_implementation_patterns"
    style="color: #9558B2;">Patterns</a>           &nbsp;|&nbsp;
  <a href="reference"
    style="color: #9558B2;">Reference</a>
</div>

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
machine learning and statistics, some examples of which are listed [here](@ref
patterns). A careful design ensures algorithms implementing LearnAPI.jl can buy into
functionality, such as external performance estimates, hyperparameter optimization and
model composition, provided by ML/statistics toolboxes and other packages. LearnAPI.jl
includes a number of Julia [traits](@ref traits) for promising specific behavior.


## Sample workflow

Suppose `forest` is some object encapsulating the hyperparameters of the [random forest
algorithm](https://en.wikipedia.org/wiki/Random_forest) (the number of trees, etc.). Then,
a LearnAPI.jl interface can be implemented, for objects with the type of `forest`, to
enable the basic workflow below. In this case data is presented following the
"scikit-learn" `X, y` pattern, although LearnAPI.jl supports other data patterns.

```julia
# `X` is some training features
# `y` is some training target
# `Xnew` is some test or production features

# List LearnaAPI functions implemented for `forest`:
@functions forest

# Train:
model = fit(forest, (X, y))

# Generate point predictions:
ŷ = predict(model, Xnew) # or `predict(model, Point(), Xnew)`

# Predict probability distributions:
predict(model, Distribution(), Xnew)

# Apply an "accessor function" to inspect byproducts of training:
LearnAPI.feature_importances(model)

# Slim down and otherwise prepare model for serialization:
small_model = LearnAPI.strip(model)
serialize("my_random_forest.jls", small_model)
```

`Distribution` and `Point` are singleton types owned by LearnAPI.jl. They allow
dispatch based on the [kind of target proxy](@ref proxy), a key LearnAPI.jl concept.
LearnAPI.jl places more emphasis on the notion of target variables and target proxies than
on the usual supervised/unsupervised learning dichotomy. From this point of view, a
supervised learner is simply one in which a target variable exists, and happens to
appear as an input to training but not to prediction.

## Data interfaces and front ends

Algorithms are free to consume data in any format. However, a method called [`obs`](@ref
data_interface) (read as "observations") gives developers the option of providing a
separate data front end for their algorithms. In this case `obs` gives users and
meta-algorithms access to an algorithm-specific representation of input data, which is
additionally guaranteed to implement a standard interface for accessing individual
observations, unless the algorithm explicitly opts out. Moreover, the `fit` and `predict`
methods can directly consume these alternative data representations, for performance
benefits in some situations, such as cross-validation.

The fallback data interface is the [MLCore.jl](https://github.com/JuliaML/MLCore.jl)
`getobs/numobs` interface (previously provided by MLUtils.jl) here tagged as
[`LearnAPI.RandomAccess()`](@ref). However, if the input consumed by the algorithm already
implements that interface (tables, arrays, etc.)  then overloading `obs` is completely
optional. Plain iteration interfaces, with or without knowledge of the number of
observations, can also be specified, to support, e.g., data loaders reading images from
disk.

Some canned data front ends (implementations of [`obs`](@ref)) are provided by the
[LearnDataFrontEnds.jl](https://juliaai.github.io/LearnDataFrontEnds.jl/stable/) package.

## Learning more

- [Anatomy of an Implementation](@ref): informal tutorial introducing the main actors in a
  new LearnAPI.jl implementation, including a **Quick Start** for new implementations.

- [Reference](@ref reference): official specification

- [Common Implementation Patterns](@ref patterns): implementation suggestions for common,
  informally defined, algorithm types

- [Testing an Implementation](@ref)
