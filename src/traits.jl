# There are two types of traits - ordinary traits that an implementation overloads to make
# promises of learner behavior, and derived traits, which are never overloaded.

const DOC_UNKNOWN =
    "Returns `\"unknown\"` if the learner implementation has "*
    "not overloaded the trait. "
const DOC_ON_TYPE = "The value of the trait must depend only on the type of `learner`. "

# const DOC_EXPLAIN_EACHOBS =
#     """

#     Here, "for each `o` in `observations`" is understood in the sense of the data
#     interface specified for the learner, [`LearnAPI.data_interface(learner)`](@ref). For
#     example, if this is `LearnAPI.RandomAccess()`, then this means "for `o` in
#     `MLCore.eachobs(observations)`".

#     """

# # OVERLOADABLE TRAITS

"""
    Learn.API.constructor(learner)

Return a keyword constructor that can be used to clone `learner`:

```julia-repl
julia> learner.lambda
0.1
julia> C = LearnAPI.constructor(learner)
julia> learner2 = C(lambda=0.2)
julia> learner2.lambda
0.2
```

# New implementations

All new implementations must overload this trait.

Attach public LearnAPI.jl-related documentation for learner to the constructor, not
the learner struct.

It must be possible to recover learner from the constructor returned as follows:

```julia
properties = propertynames(learner)
named_properties = NamedTuple{properties}(getproperty.(Ref(learner), properties))
@assert learner == LearnAPI.constructor(learner)(; named_properties...)
```

which can be tested with `@assert LearnAPI.clone(learner) == learner`.

The keyword constructor provided by `LearnAPI.constructor` must provide default values for
all properties, with the exception of those that can take other LearnAPI.jl learners as
values. These can be provided with the default `nothing`, with the constructor throwing an
error if the default value persists.

"""
function constructor end

"""
    LearnAPI.functions(learner)

Return a tuple of expressions representing functions that can be meaningfully applied with
`learner`, or an associated model (object returned by `fit(learner, ...)`), as the first
argument. Learner traits (methods for which `learner` is the *only* argument) are
excluded.

To return actual functions, instead of symbols, use [`@functions`](@ref)  `learner`
instead.

The returned tuple may include expressions like `:(DecisionTree.print_tree)`, which
reference functions not owned by LearnAPI.jl.

The understanding is that `learner` is a LearnAPI-compliant object whenever the return
value is non-empty.

Do `LearnAPI.functions()` to list all possible elements of the return value representing
functions owned by LearnAPI.jl.

# Extended help

# New implementations

All new implementations must implement this trait. Here's a checklist for elements in the
return value:

| expression                        | implementation compulsory? | include in returned tuple?       |
|:----------------------------------|:---------------------------|:---------------------------------|
| `:(LearnAPI.fit)`                 | yes                        | yes                              |
| `:(LearnAPI.learner)`             | yes                        | yes                              |
| `:(LearnAPI.clone)`               | never overloaded           | yes                              |
| `:(LearnAPI.strip)`               | no                         | yes                              |
| `:(LearnAPI.obs)`                 | no                         | yes                              |
| `:(LearnAPI.features)`            | no                         | yes, unless `learner` is static  |
| `:(LearnAPI.target)`              | no                         | only if implemented              |
| `:(LearnAPI.weights)`             | no                         | only if implemented              |
| `:(LearnAPI.update)`              | no                         | only if implemented              |
| `:(LearnAPI.update_observations)` | no                         | only if implemented              |
| `:(LearnAPI.update_features)`     | no                         | only if implemented              |
| `:(LearnAPI.predict)`             | no                         | only if implemented              |
| `:(LearnAPI.transform)`           | no                         | only if implemented              |
| `:(LearnAPI.inverse_transform)`   | no                         | only if implemented              |
| < accessor functions>             | no                         | only if implemented              |

Also include any implemented accessor functions, both those owned by LearnaAPI.jl, and any
learner-specific ones. The LearnAPI.jl accessor functions are: $ACCESSOR_FUNCTIONS_LIST
(`LearnAPI.strip` is always included).

"""
functions(::Any) = ()
functions() = (
    :(LearnAPI.fit),
    :(LearnAPI.learner),
    :(LearnAPI.clone),
    :(LearnAPI.strip),
    :(LearnAPI.obs),
    :(LearnAPI.features),
    :(LearnAPI.target),
    :(LearnAPI.weights),
    :(LearnAPI.update),
    :(LearnAPI.update_observations),
    :(LearnAPI.update_features),
    :(LearnAPI.predict),
    :(LearnAPI.transform),
    :(LearnAPI.inverse_transform),
    ACCESSOR_FUNCTIONS...,
)

"""
    @functions learner

Return a tuple of functions that can be meaningfully applied with `learner`, or an
associated model, as the first argument. An "associated model" is an object returned by
`fit(learner, ...)`. Learner traits (methods for which `learner` always the *only*
argument) are excluded.

```
julia> @functions my_feature_selector
(fit, LearnAPI.learner, clone, strip, obs, transform)

```

New learner implementations should overload [`LearnAPI.functions`](@ref).

See also [`LearnAPI.functions`](@ref).

"""
macro functions(learner)
    quote
        exs = LearnAPI.functions(learner)
        eval.(exs)
    end |> esc
end

"""
    LearnAPI.kinds_of_proxy(learner)

Returns a tuple of all instances, `kind`, for which for which `predict(learner, kind,
data...)` has a guaranteed implementation. Each such `kind` subtypes
[`LearnAPI.KindOfProxy`](@ref). Examples are `Point()` (for predicting actual
target values) and `Distributions()` (for predicting probability mass/density functions).

The call `predict(model, data)` always returns `predict(model, kind, data)`, where `kind`
is the first element of the trait's return value.

See also [`LearnAPI.predict`](@ref), [`LearnAPI.KindOfProxy`](@ref).

# Extended help

# New implementations

Must be overloaded whenever `predict` is implemented.

Elements of the returned tuple must be instances of [`LearnAPI.KindOfProxy`](@ref). List
all possibilities by running `LearnAPI.kinds_of_proxy()`.

Suppose, for example, we have the following implementation of a supervised learner
returning only probabilistic predictions:

```julia
LearnAPI.predict(learner::MyNewLearnerType, LearnAPI.Distribution(), Xnew) = ...
```

Then we can declare

```julia
@trait MyNewLearnerType kinds_of_proxy = (LearnaAPI.Distribution(),)
```

LearnAPI.jl provides the fallback for `predict(model, data)`.

For more on target variables and target proxies, refer to the LearnAPI documentation.

"""
kinds_of_proxy(::Any) = ()
kinds_of_proxy() = map(CONCRETE_TARGET_PROXY_SYMBOLS) do ex
    quote
        $ex()
    end |> eval
end

tags() = [
    "regression",
    "classification",
    "clustering",
    "gradient descent",
    "iterative algorithms",
    "incremental algorithms",
    "feature engineering",
    "dimension reduction",
    "missing value imputation",
    "transformers",
    "static algorithms",
    "ensembling",
    "time series forecasting",
    "time series classification",
    "survival analysis",
    "density estimation",
    "Bayesian algorithms",
    "outlier detection",
    "collaborative filtering",
    "text analysis",
    "audio analysis",
    "natural language processing",
    "image processing",
    "meta-algorithms"
]

"""
    LearnAPI.tags(learner)

Lists one or more suggestive learner tags. Do `LearnAPI.tags()` to list
all possible.

!!! warning
    The value of this trait guarantees no particular behavior. The trait is
    intended for informal classification purposes only.

# New implementations

This trait should return a tuple of strings, as in `("classifier", "text analysis")`.

"""
tags(::Any) = ()

"""
    LearnAPI.is_pure_julia(learner)

Returns `true` if training `learner` requires evaluation of pure Julia code only.

# New implementations

The fallback is `false`.

"""
is_pure_julia(::Any) = false

"""
    LearnAPI.pkg_name(learner)

Return the name of the package module which supplies the core training algorithm for
`learner`.  This is not necessarily the package providing the LearnAPI
interface.

$DOC_UNKNOWN

# New implementations

Must return a string, as in `"DecisionTree"`.

"""
pkg_name(::Any) = "unknown"

"""
    LearnAPI.pkg_license(learner)

Return the name of the software license, such as `"MIT"`, applying to the package where the
core algorithm for `learner` is implemented.

"""
pkg_license(::Any) = "unknown"

"""
    LearnAPI.doc_url(learner)

Return a url where the core algorithm for `learner` is documented.

$DOC_UNKNOWN

# New implementations

Must return a string, such as `"https://en.wikipedia.org/wiki/Decision_tree_learning"`.

"""
doc_url(::Any) = "unknown"

"""
    LearnAPI.load_path(learner)

Return a string indicating where in code the definition of the learner's constructor can
be found, beginning with the name of the package module defining it. By "constructor" we
mean the return value of [`LearnAPI.constructor(learner)`](@ref).

# Implementation

For example, a return value of `"FastTrees.LearnAPI.DecisionTreeClassifier"` means the
following julia code will not error:

```julia
import FastTrees
import LearnAPI
@assert FastTrees.LearnAPI.DecisionTreeClassifier == LearnAPI.constructor(learner)
```

$DOC_UNKNOWN


"""
load_path(::Any) = "unknown"


"""
    LearnAPI.nonlearners(learner)

Return the properties of `learner` whose corresponding values are not themselves
learners.

See also [`LearnAPI.learners`](@ref).

# New implementations

This trait should be overloaded if one or more properties (fields) of `learner` take
learner values. The fallback returns `propertynames(learner)`, meaning no properties have
learner values. If overloaded, implementation of the accessor function
[`LearnAPI.components`](@ref) is recommended.

$DOC_ON_TYPE


"""
nonlearners(learner) = propertynames(learner)

"""
    LearnAPI.human_name(learner)

Return a human-readable string representation of `typeof(learner)`. Primarily intended
for auto-generation of documentation.

# New implementations

Optional. A fallback takes the type name, inserts spaces and removes capitalization. For
example, `KNNRegressor` becomes `"knn regressor"`. Better would be to overload the trait
to return `"K-nearest neighbors regressor"`. Ideally, this is a "concrete" noun like
`"ridge regressor"` rather than an "abstract" noun like `"ridge regression"`.

"""
human_name(learner) = snakecase(name(learner), delim=' ') # `name` defined below

"""
    LearnAPI.data_interface(learner)

Return the data interface supported by `learner` for accessing individual observations
in representations of input data returned by [`obs(learner, data)`](@ref) or
[`obs(model, data)`](@ref), whenever `learner == LearnAPI.learner(model)`. Here `data`
is `fit`, `predict`, or `transform`-consumable data.

See [`LearnAPI.DataInterface`](@ref) for possible return values.

See also [`obs`](@ref).

# New implementations

The fallback returns [`LearnAPI.RandomAccess`](@ref), which applies to arrays, most
tables, and tuples of these. See the doc-string for details.

"""
data_interface(::Any) = LearnAPI.RandomAccess()

"""
    LearnAPI.is_static(learner)

Returns `true` if [`fit`](@ref) is called with no data arguments, as in
`fit(learner)`. That is, `learner` does not generalize to new data, and data is only
provided at the `predict` or `transform` step.

For example, some clustering algorithms are applied with this workflow, to assign labels
to the observations in `X`:

```julia
model = fit(learner) # no training data
labels = predict(model, X) # may mutate `model`!

# extract some byproducts of the clustering algorithm (e.g., outliers):
LearnAPI.extras(model)
```

# New implementations

This trait, falling back to `false`, may only be overloaded when `fit` has no data
arguments. See more at [`fit`](@ref).

"""
is_static(::Any) = false

"""
    LearnAPI.iteration_parameter(learner)

The name of the iteration parameter of `learner`, or `nothing` if the algorithm is not
iterative.

# New implementations

Implement if algorithm is iterative. Returns a symbol or `nothing`.

"""
iteration_parameter(::Any) = nothing

# """
#     LearnAPI.fit_observation_scitype(learner)

# Return an upper bound `S` on the scitype of individual observations guaranteed to work
# when calling `fit`: if `observations = obs(learner, data)` and
# `ScientificTypes.scitype(collect(o)) <:S` for each `o` in `observations`, then the call
# `fit(learner, data)` is supported.

# $DOC_EXPLAIN_EACHOBS

# See also [`LearnAPI.target_observation_scitype`](@ref).

# # New implementations

# Optional. The fallback return value is `Union{}`.

# """
# fit_observation_scitype(::Any) = Union{}

"""
    LearnAPI.fit_scitype(learner)

Return an upper bound `S` on the `scitype` (scientific type) of `data` for which the call
[`fit(learner, data)`](@ref) is supported. Specifically, if `ScientificTypes.scitype(data)
<: S` then the call is guaranteed to succeed. If not, the call may or may not succeed.

See ScientificTypes.jl documentation for more on the `scitype` function.

See also [`LearnAPI.target_observation_scitype`](@ref).

# New implementations

Optional. The fallback return value is `Union{}`.

"""
fit_scitype(::Any) = Union{}

"""
    LearnAPI.target_observation_scitype(learner)

Return an upper bound `S` on the `scitype` (scientific type) of each observation of any
target variable associated with the learner. See LearnAPI.jl documentation for the meaning
of "target variable".  See ScientificTypes.jl documentation for an explanation of the
`scitype` function, which it provides.

Specifically, both of the following are always true:

- If `:(LearnAPI.target) in LearnAPI.functions(learner)` (i.e., `fit` consumes target
  variables) then `ScientificTypes.scitype(o) <: S` for each `o` in `target_observations`,
  where `target_observations = `[`LearnAPI.target(learner, observations)`](@ref),
  `observations = `[`LearnAPI.obs(learner, data)`](@ref), and `data` is a supported
  argument in the call [`fit(learner, data)`](@ref).  Here, "for each `o` in
  `target_observations`" is understood in the sense of the data interface specified for
  the learner, [`LearnAPI.data_interface(learner)`](@ref). For example, if this is
  `LearnAPI.RandomAccess()`, then this means "for each `o in
  MLCore.eachobs(target_observations)`".

- `S` is an upper bound on the `scitype` of (point) observations that might normally be
  extracted from the output of [`predict`](@ref).

To illustate the second property, suppose we have

```julia
model = fit(learner, data)
ŷ = predict(model, Sampleable(), data_new)
```

Then each individual sample generated by each "observation" of `ŷ` (a vector of sampleable
objects, say) will be bound in `scitype` by `S`.

See also See also [`LearnAPI.fit_scitype`](@ref).

# New implementations

Optional. The fallback return value is `Any`.

"""
target_observation_scitype(::Any) = Any


# # DERIVED TRAITS

name(learner) = split(string(constructor(learner)), ".") |> last

"""
    LearnAPI.learners(learner)

Return the properties of `learner` whose corresponding values are themselves
learners.

See also [`LearnAPI.learners`](@ref).

# New implementations

This trait should not be overloaded. Instead overload [`LearnAPI.nonlearners`](@ref).

"""
learners(learner) = setdiff(propertynames(learner), nonlearners(learner))

"""
    LearnAPI.is_learner(object)

Returns `true` if `object` has a valid implementation of the LearnAPI.jl
interface. Equivalent to non-emptiness of [`LearnAPI.functions(object)`](@ref).

This trait should never be overloaded explicitly.

"""
is_learner(learner) = !isempty(functions(learner))
preferred_kind_of_proxy(learner) = first(kinds_of_proxy(learner))
target(learner) = :(LearnAPI.target) in functions(learner)
weights(learner) = :(LearnAPI.weights) in functions(learner)
