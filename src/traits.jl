# There are two types of traits - ordinary traits that an implementation overloads to make
# promises of algorithm behavior, and derived traits, which are never overloaded.

const DOC_UNKNOWN =
    "Returns `\"unknown\"` if the algorithm implementation has "*
    "not overloaded the trait. "
const DOC_ON_TYPE = "The value of the trait must depend only on the type of `algorithm`. "

DOC_ONLY_ONE(func) =
    "Ordinarily, at most one of the following should be overloaded for given "*
    "algorithm "*
    "`LearnAPI.$(func)_scitype`, `LearnAPI.$(func)_type`, "*
    "`LearnAPI.$(func)_observation_scitype`, "*
    "`LearnAPI.$(func)_observation_type`."

const DOC_EXPLAIN_EACHOBS =
    """

    Here, "for each `o` in `observations`" is understood in the sense of
    [`LearnAPI.data_interface(algorithm)`](@ref). For example, if
    `LearnAPI.data_interface(algorithm) == Base.HasLength()`, then this means "for `o` in
    `MLUtils.eachobs(observations)`".

    """

# # OVERLOADABLE TRAITS

"""
    Learn.API.constructor(algorithm)

Return a keyword constructor that can be used to clone `algorithm`:

```julia-repl
julia> algorithm.lambda
0.1
julia> C = LearnAPI.constructor(algorithm)
julia> algorithm2 = C(lambda=0.2)
julia> algorithm2.lambda
0.2
```

# New implementations

All new implementations must overload this trait.

Attach public LearnAPI.jl-related documentation for an algorithm to the constructor, not
the algorithm struct.

It must be possible to recover an algorithm from the constructor returned as follows:

```julia
properties = propertynames(algorithm)
named_properties = NamedTuple{properties}(getproperty.(Ref(algorithm), properties))
@assert algorithm == LearnAPI.constructor(algorithm)(; named_properties...)
```

which can be tested with `@assert LearnAPI.clone(algorithm) == algorithm`.

The keyword constructor provided by `LearnAPI.constructor` must provide default values for
all properties, with the exception of those that can take other LearnAPI.jl algorithms as
values. These can be provided with the default `nothing`, with the constructor throwing an
error if the default value persists.

"""
function constructor end

"""
    LearnAPI.functions(algorithm)

Return a tuple of expressions representing functions that can be meaningfully applied
with `algorithm`, or an associated model (object returned by `fit(algorithm, ...)`, as the
first argument. Algorithm traits (methods for which `algorithm` is the *only* argument)
are excluded.

The returned tuple may include expressions like `:(DecisionTree.print_tree)`, which
reference functions not owned by LearnAPI.jl.

The understanding is that `algorithm` is a LearnAPI-compliant object whenever the return
value is non-empty.

# Extended help

# New implementations

All new implementations must overload this trait. Here's a checklist for elements in the
return value:

| expression                        | implementation compulsory? | include in returned tuple?         |
|-----------------------------------|----------------------------|------------------------------------|
| `:(LearnAPI.fit)`                 | yes                        | yes                                |
| `:(LearnAPI.algorithm)`           | yes                        | yes                                |
| `:(LearnAPI.minimize)`            | no                         | yes                                |
| `:(LearnAPI.obs)`                 | no                         | yes                                |
| `:(LearnAPI.features)`            | no                         | yes, unless `fit` consumes no data |
| `:(LearnAPI.target)`              | no                         | only if implemented                |
| `:(LearnAPI.weights)`             | no                         | only if implemented                |
| `:(LearnAPI.update)`              | no                         | only if implemented                |
| `:(LearnAPI.update_observations)` | no                         | only if implemented                |
| `:(LearnAPI.update_features)`     | no                         | only if implemented                |
| `:(LearnAPI.predict)`             | no                         | only if implemented                |
| `:(LearnAPI.transform)`           | no                         | only if implemented                |
| `:(LearnAPI.inverse_transform)`   | no                         | only if implemented                |
| <accessor functions>              | no                         | only if implemented                |

Also include any implemented accessor functions, both those owned by LearnaAPI.jl, and any
algorithm-specific ones. The LearnAPI.jl accessor functions are: $ACCESSOR_FUNCTIONS_LIST.

"""
functions(::Any) = ()
functions() = (
    :(LearnAPI.fit),
    :(LearnAPI.algorithm),
    :(LearnAPI.minimize),
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
)

"""
    LearnAPI.kinds_of_proxy(algorithm)

Returns a tuple of all instances, `kind`, for which for which `predict(algorithm, kind,
data...)` has a guaranteed implementation. Each such `kind` subtypes
[`LearnAPI.KindOfProxy`](@ref). Examples are `Point()` (for predicting actual
target values) and `Distributions()` (for predicting probability mass/density functions).

The call `predict(model, data)` always returns `predict(model, kind, data)`, where `kind`
is the first element of the trait's return value.

See also [`LearnAPI.predict`](@ref), [`LearnAPI.KindOfProxy`](@ref).

# Extended help

# New implementations

Must be overloaded whenever `predict` is implemented.

Elements of the returned tuple must be instances of types in the return value of
`LearnAPI.kinds_of_proxy()`, i.e., one of the following, described further in LearnAPI.jl
documentation: $CONCRETE_TARGET_PROXY_TYPES_LIST.

Suppose, for example, we have the following implementation of a supervised learner
returning only probabilistic predictions:

```julia
LearnAPI.predict(algorithm::MyNewAlgorithmType, LearnAPI.Distribution(), Xnew) = ...
```

Then we can declare

```julia
@trait MyNewAlgorithmType kinds_of_proxy = (LearnaAPI.Distribution(),)
```

LearnAPI.jl provides the fallback for `predict(model, data)`.

For more on target variables and target proxies, refer to the LearnAPI documentation.

"""
kinds_of_proxy(::Any) = ()
kinds_of_proxy() = CONCRETE_TARGET_PROXY_TYPES


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
    "ensemble algorithms",
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

const DOC_TAGS_LIST = join(map(d -> "`\"$d\"`", tags()), ", ")

"""
    LearnAPI.tags(algorithm)

Lists one or more suggestive algorithm tags. Do `LearnAPI.tags()` to list
all possible.

!!! warning
    The value of this trait guarantees no particular behavior. The trait is
    intended for informal classification purposes only.

# New implementations

This trait should return a tuple of strings, as in `("classifier", "text analysis")`.

"""
tags(::Any) = ()

"""
    LearnAPI.is_pure_julia(algorithm)

Returns `true` if training `algorithm` requires evaluation of pure Julia code only.

# New implementations

The fallback is `false`.

"""
is_pure_julia(::Any) = false

"""
    LearnAPI.pkg_name(algorithm)

Return the name of the package module which supplies the core training algorithm for
`algorithm`.  This is not necessarily the package providing the LearnAPI
interface.

$DOC_UNKNOWN

# New implementations

Must return a string, as in `"DecisionTree"`.

"""
pkg_name(::Any) = "unknown"

"""
    LearnAPI.pkg_license(algorithm)

Return the name of the software license, such as `"MIT"`, applying to the package where the
core algorithm for `algorithm` is implemented.

"""
pkg_license(::Any) = "unknown"

"""
    LearnAPI.doc_url(algorithm)

Return a url where the core algorithm for `algorithm` is documented.

$DOC_UNKNOWN

# New implementations

Must return a string, such as `"https://en.wikipedia.org/wiki/Decision_tree_learning"`.

"""
doc_url(::Any) = "unknown"

"""
    LearnAPI.load_path(algorithm)

Return a string indicating where in code the definition of the algorithm's constructor can
be found, beginning with the name of the package module defining it. By "constructor" we
mean the return value of [`LearnAPI.constructor(algorithm)`](@ref).

# Implementation

For example, a return value of `"FastTrees.LearnAPI.DecisionTreeClassifier"` means the
following julia code will not error:

```julia
import FastTrees
import LearnAPI
@assert FastTrees.LearnAPI.DecisionTreeClassifier == LearnAPI.constructor(algorithm)
```

$DOC_UNKNOWN


"""
load_path(::Any) = "unknown"


"""
    LearnAPI.is_composite(algorithm)

Returns `true` if one or more properties (fields) of `algorithm` may themselves be
algorithms, and `false` otherwise.

See also [`LearnAPI.components`](@ref).

# New implementations

This trait should be overloaded if one or more properties (fields) of `algorithm` may take
algorithm values. Fallback return value is `false`. The keyword constructor for such an
algorithm need not prescribe defaults for algorithm-valued properties. Implementation of
the accessor function [`LearnAPI.components`](@ref) is recommended.

$DOC_ON_TYPE


"""
is_composite(::Any) = false

"""
    LearnAPI.human_name(algorithm)

Return a human-readable string representation of `typeof(algorithm)`. Primarily intended
for auto-generation of documentation.

# New implementations

Optional. A fallback takes the type name, inserts spaces and removes capitalization. For
example, `KNNRegressor` becomes `"knn regressor"`. Better would be to overload the trait
to return `"K-nearest neighbors regressor"`. Ideally, this is a "concrete" noun like
`"ridge regressor"` rather than an "abstract" noun like `"ridge regression"`.

"""
human_name(algorithm) = snakecase(name(algorithm), delim=' ') # `name` defined below

"""
    LearnAPI.data_interface(algorithm)

Return the data interface supported by `algorithm` for accessing individual observations
in representations of input data returned by [`obs(algorithm, data)`](@ref) or
[`obs(model, data)`](@ref), whenever `algorithm == LearnAPI.algorithm(model)`. Here `data`
is `fit`, `predict`, or `transform`-consumable data.

Possible return values are [`LearnAPI.RandomAccess`](@ref),
[`LearnAPI.FiniteIterable`](@ref), and [`LearnAPI.Iterable`](@ref).

See also [`obs`](@ref).

# New implementations

The fallback returns [`LearnAPI.RandomAccess`](@ref), which applies to arrays, most
tables, and tuples of these. See the doc-string for details.

"""
data_interface(::Any) = LearnAPI.RandomAccess()

"""
    LearnAPI.is_static(algorithm)

Returns `true` if [`fit`](@ref) is called with no data arguments, as in
`fit(algorithm)`. That is, `algorithm` does not generalize to new data, and data is only
provided at the `predict` or `transform` step.

For example, some clustering algorithms are applied with this workflow, to assign labels
to the observations in `X`:

```julia
model = fit(algorithm) # no training data
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
    LearnAPI.iteration_parameter(algorithm)

The name of the iteration parameter of `algorithm`, or `nothing` if the algorithm is not
iterative.

# New implementations

Implement if algorithm is iterative. Returns a symbol or `nothing`.

"""
iteration_parameter(::Any) = nothing


"""
    LearnAPI.fit_observation_scitype(algorithm)

Return an upper bound `S` on the scitype of individual observations guaranteed to work
when calling `fit`: if `observations = obs(algorithm, data)` and
`ScientificTypes.scitype(o) <:S` for each `o` in `observations`, then the call
`fit(algorithm, data)` is supported.

$DOC_EXPLAIN_EACHOBS

See also [`LearnAPI.target_observation_scitype`](@ref).

# New implementations

Optional. The fallback return value is `Union{}`. $(DOC_ONLY_ONE(:fit))

"""
fit_observation_scitype(::Any) = Union{}

"""
    LearnAPI.target_observation_scitype(algorithm)

Return an upper bound `S` on the scitype of each observation of an applicable target
variable. Specifically:

- If `:(LearnAPI.target) in LearnAPI.functions(algorithm)` (i.e., `fit` consumes target
  variables) then "target" means anything returned by `LearnAPI.target(algorithm, data)`,
  where `data` is an admissible argument in the call `fit(algorithm, data)`.

- `S` will always be an upper bound on the scitype of (point) observations that could be
  conceivably extracted from the output of [`predict`](@ref).

To illustate the second case, suppose we have

```julia
model = fit(algorithm, data)
ŷ = predict(model, Sampleable(), data_new)
```

Then each individual sample generated by each "observation" of `ŷ` (a vector of sampleable
objects, say) will be bound in scitype by `S`.

See also See also [`LearnAPI.fit_observation_scitype`](@ref).

# New implementations

Optional. The fallback return value is `Any`.

"""
target_observation_scitype(::Any) = Any


# # DERIVED TRAITS

name(algorithm) = split(string(constructor(algorithm)), ".") |> last
is_algorithm(algorithm) = !isempty(functions(algorithm))
preferred_kind_of_proxy(algorithm) = first(kinds_of_proxy(algorithm))
target(algorithm) = :(LearnAPI.target) in functions(algorithm)
weights(algorithm) = :(LearnAPI.weights) in functions(algorithm)
