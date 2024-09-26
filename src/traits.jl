# There are two types of traits - ordinary traits that an implementation overloads to make
# promises of algorithm behavior, and derived traits, which are never overloaded.

const DOC_UNKNOWN =
    "Returns `\"unknown\"` if the algorithm implementation has "*
    "failed to overload the trait. "
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

const TRAITS = [
    :constructor,
    :functions,
    :kinds_of_proxy,
    :descriptors,
    :is_pure_julia,
    :pkg_name,
    :pkg_license,
    :doc_url,
    :load_path,
    :is_composite,
    :human_name,
    :iteration_parameter,
    :data_interface,
    :predict_or_transform_mutates,
    :fit_scitype,
    :fit_observation_scitype,
    :fit_type,
    :fit_observation_type,
    :target_observation_scitype,
    :predict_input_scitype,
    :predict_output_scitype,
    :predict_input_type,
    :predict_output_type,
    :transform_input_scitype,
    :transform_output_scitype,
    :transform_input_type,
    :transform_output_type,
    :name,
    :is_algorithm,
]


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

The keyword constructor provided by `LearnAPI.constructor` must provide default values for
all properties, with the exception of those that can take other LearnAPI.jl algorithms as
values.

"""
function constructor end

"""
    LearnAPI.functions(algorithm)

Return a tuple of expressions respresenting functions that can be meaningfully applied
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

| symbol                          | implementation/overloading compulsory? | include in returned tuple?         |
|---------------------------------|----------------------------------------|------------------------------------|
| `:(LearnAPI.fit)`               | yes                                    | yes                                |
| `:(LearnAPI.algorithm)`         | yes                                    | yes                                |
| `:(LearnAPI.minimize)`          | no                                     | yes                                |
| `:(LearnAPI.obs)`               | no                                     | yes                                |
| `:(LearnAPI.input)`             | no                                     | yes, unless `fit` consumes no data |
| `:(LearnAPI.target)`            | no                                     | only if implemented                |
| `:(LearnAPI.weights)`           | no                                     | only if implemented                |
| `:(LearnAPI.predict)`           | no                                     | only if implemented                |
| `:(LearnAPI.transform)`         | no                                     | only if implemented                |
| `:(LearnAPI.inverse_transform)` | no                                     | only if implemented                |
| <accessor functions>            | no                                     | only if implemented                |

Also include any implemented accessor functions, both those owned by LearnaAPI.jl, and any
algorithm-specific ones. The LearnAPI.jl accessor functions are: $ACCESSOR_FUNCTIONS_LIST.

"""
functions(::Any) = ()


"""
    LearnAPI.kinds_of_proxy(algorithm)

Returns a tuple of all instances, `kind`, for which for which `predict(algorithm, kind,
data...)` has a guaranteed implementation. Each such `kind` subtypes
[`LearnAPI.KindOfProxy`](@ref). Examples are `LiteralTarget()` (for predicting actual
target values) and `Distributions()` (for predicting probability mass/density functions).

If a `predict(model, data)` is overloaded to return predictions for a specific kind of
proxy (e.g., `predict(model::MyModel, data) = predict(model, Distribution(), data)`) then
that kind appears first in the returned tuple.

See also [`LearnAPI.predict`](@ref), [`LearnAPI.KindOfProxy`](@ref).

# Extended help

# New implementations

Implementation is optional but recommended whenever `predict` is overloaded.

Elements of the returned tuple must be one of these: $CONCRETE_TARGET_PROXY_TYPES_LIST.

Suppose, for example, we have the following implementation of a supervised learner
returning only probabilistic predictions:

```julia
LearnAPI.predict(algorithm::MyNewAlgorithmType, LearnAPI.Distribution(), Xnew) = ...
```

Then we can declare

```julia
@trait MyNewAlgorithmType kinds_of_proxy = (LearnaAPI.Distribution(),)
```

For more on target variables and target proxies, refer to the LearnAPI documentation.

"""
kinds_of_proxy(::Any) = ()

descriptors() = [
    :regression,
    :classification,
    :clustering,
    :gradient_descent,
    :iterative_algorithms,
    :incremental_algorithms,
    :dimension_reduction,
    :encoders,
    :static_algorithms,
    :missing_value_imputation,
    :ensemble_algorithms,
    :wrappers,
    :time_series_forecasting,
    :time_series_classification,
    :survival_analysis,
    :distribution_fitters,
    :Bayesian_algorithms,
    :outlier_detection,
    :collaborative_filtering,
    :text_analysis,
    :audio_analysis,
    :natural_language_processing,
    :image_processing,
]

const DOC_DESCRIPTORS_LIST = join(map(d -> "`:$d`", descriptors()), ", ")

"""
    LearnAPI.descriptors(algorithm)

Lists one or more suggestive algorithm descriptors from this list: $DOC_DESCRIPTORS_LIST (do
`LearnAPI.descriptors()` to reproduce).

!!! warning
    The value of this trait guarantees no particular behavior. The trait is
    intended for informal classification purposes only.

# New implementations

This trait should return a tuple of symbols, as in `(:classifier, :text_analysis)`.

"""
descriptors(::Any) = ()

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
human_name(M) = snakecase(name(M), delim=' ') # `name` defined below

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
    LearnAPI.predict_or_transform_mutates(algorithm)

Returns `true` if [`predict`](@ref) or [`transform`](@ref) possibly mutate their first
argument, `model`, when `LearnAPI.algorithm(model) == algorithm`. If `false`, no arguments
are ever mutated.

# New implementations

This trait, falling back to `false`, may only be overloaded when `fit` has no data
arguments (`algorithm` does not generalize to new data). See more at [`fit`](@ref).

"""
predict_or_transform_mutates(::Any) = false

"""
    LearnAPI.iteration_parameter(algorithm)

The name of the iteration parameter of `algorithm`, or `nothing` if the algorithm is not
iterative.

# New implementations

Implement if algorithm is iterative. Returns a symbol or `nothing`.

"""
iteration_parameter(::Any) = nothing


"""
    LearnAPI.fit_scitype(algorithm)

Return an upper bound `S` on the scitype of `data` guaranteed to work when calling
`fit(algorithm, data)`: if `ScientificTypes.scitype(data) <: S`, then is `fit(algorithm,
data)` is supported.

See also [`LearnAPI.fit_type`](@ref), [`LearnAPI.fit_observation_scitype`](@ref),
[`LearnAPI.fit_observation_type`](@ref).

# New implementations

Optional. The fallback return value is `Union{}`.  $(DOC_ONLY_ONE(:fit))

"""
fit_scitype(::Any) = Union{}

"""
    LearnAPI.fit_observation_scitype(algorithm)

Return an upper bound `S` on the scitype of individual observations guaranteed to work
when calling `fit`: if `observations = obs(algorithm, data)` and
`ScientificTypes.scitype(o) <:S` for each `o` in `observations`, then the call
`fit(algorithm, data)` is supported.

$DOC_EXPLAIN_EACHOBS

See also See also [`LearnAPI.fit_type`](@ref), [`LearnAPI.fit_scitype`](@ref),
[`LearnAPI.fit_observation_type`](@ref).

# New implementations

Optional. The fallback return value is `Union{}`. $(DOC_ONLY_ONE(:fit))

"""
fit_observation_scitype(::Any) = Union{}

"""
    LearnAPI.fit_type(algorithm)

Return an upper bound `T` on the type of `data` guaranteed to work when calling
`fit(algorithm, data)`: if `typeof(data) <: T`, then `fit(algorithm, data)` is supported.

See also [`LearnAPI.fit_scitype`](@ref), [`LearnAPI.fit_observation_type`](@ref).
[`LearnAPI.fit_observation_scitype`](@ref)

# New implementations

Optional. The fallback return value is `Union{}`. $(DOC_ONLY_ONE(:fit))

"""
fit_type(::Any) = Union{}

"""
    LearnAPI.fit_observation_type(algorithm)

Return an upper bound `T` on the type of individual observations guaranteed to work
when calling `fit`: if `observations = obs(algorithm, data)` and
`typeof(o) <:S` for each `o` in `observations`, then the call
`fit(algorithm, data)` is supported.

$DOC_EXPLAIN_EACHOBS

See also See also [`LearnAPI.fit_type`](@ref), [`LearnAPI.fit_scitype`](@ref),
[`LearnAPI.fit_observation_scitype`](@ref).

# New implementations

Optional. The fallback return value is `Union{}`. $(DOC_ONLY_ONE(:fit))

"""
fit_observation_type(::Any) = Union{}

"""
    LearnAPI.target_observation_scitype(algorithm)

Return an upper bound `S` on the scitype of each observation of `LearnAPI.target(data)`,
where `data` is an admissible argument in the call `fit(algorithm, data)`.

This interpretation only holds if `LearnAPI.target(algorithm)` is `true`. In any case,
however, if `algorithm` implements `predict`, then `S` will always be an
upper bound on the scitype of observations that could be conceivably extracted from the
output of [`predict`](@ref). For example, suppose we have

```julia
model = fit(algorithm, data)
ŷ = predict(model, Sampleable(), data_new)
```

Then each sample generated by each "observation" of `ŷ` (a vector of sampleable objects,
say) will be bound in scitype by `S`.

See also See also [`LearnAPI.fit_observation_scitype`](@ref).

# New implementations

Optional. The fallback return value is `Any`.

"""
target_observation_scitype(::Any) = Any


function DOC_INPUT_SCITYPE(op)
    extra = op == :predict ? " kind_of_proxy," : ""
    ONLY = DOC_ONLY_ONE(op)
    """
        LearnAPI.$(op)_input_scitype(algorithm)

    Return an upper bound `S` on the scitype of `data` guaranteed to work in the call
    `$op(algorithm,$extra data)`: if `ScientificTypes.scitype(data) <: S`,
    then `$op(algorithm,$extra data)` is supported.

    See also [`LearnAPI.$(op)_input_type`](@ref).

    # New implementations

    Implementation is optional. The fallback return value is `Union{}`. $ONLY

   """
end

function DOC_INPUT_OBSERVATION_SCITYPE(op)
    extra = op == :predict ? " kind_of_proxy," : ""
    ONLY = DOC_ONLY_ONE(op)
    """
        LearnAPI.$(op)_observation_scitype(algorithm)

    Return an upper bound `S` on the scitype of individual observations guaranteed to work
    when calling `$op`: if `observations = obs(model, data)`, for some `model` returned by
    `fit(algorithm, ...)`, and `ScientificTypes.scitype(o) <: S` for each `o` in
    `observations`, then the call `$(op)(model,$extra data)` is supported.

    $DOC_EXPLAIN_EACHOBS

    See also See also [`LearnAPI.fit_type`](@ref), [`LearnAPI.fit_scitype`](@ref),
    [`LearnAPI.fit_observation_type`](@ref).

    # New implementations

    Optional. The fallback return value is `Union{}`. $ONLY

    """
end

function DOC_INPUT_TYPE(op)
    extra = op == :predict ? " kind_of_proxy," : ""
    ONLY = DOC_ONLY_ONE(op)
    """
        LearnAPI.$(op)_input_type(algorithm)

    Return an upper bound `T` on the scitype of `data` guaranteed to work in the call
    `$op(algorithm,$extra data)`: if `typeof(data) <: T`,
    then `$op(algorithm,$extra data)` is supported.

    See also [`LearnAPI.$(op)_input_type`](@ref).

    # New implementations

    Implementation is optional. The fallback return value is `Union{}`. Should not be
    overloaded if `LearnAPI.$(op)_input_scitype` is overloaded.

    """
end

function DOC_INPUT_OBSERVATION_TYPE(op)
    extra = op == :predict ? " kind_of_proxy," : ""
    ONLY = DOC_ONLY_ONE(op)
    """
        LearnAPI.$(op)_observation_type(algorithm)

    Return an upper bound `T` on the scitype of individual observations guaranteed to work
    when calling `$op`: if `observations = obs(model, data)`, for some `model` returned by
    `fit(algorithm, ...)`, and `typeof(o) <: T` for each `o` in
    `observations`, then the call `$(op)(model,$extra data)` is supported.

    $DOC_EXPLAIN_EACHOBS

    See also See also [`LearnAPI.fit_type`](@ref), [`LearnAPI.fit_scitype`](@ref),
    [`LearnAPI.fit_observation_type`](@ref).

    # New implementations

    Optional. The fallback return value is `Union{}`. $ONLY

    """
end

DOC_OUTPUT_SCITYPE(op) =
    """
        LearnAPI.$(op)_output_scitype(algorithm)

    Return an upper bound on the scitype of the output of the `$op` operation.

    See also [`LearnAPI.$(op)_input_scitype`](@ref).

    # New implementations

    Implementation is optional. The fallback return value is `Any`.

    """

DOC_OUTPUT_TYPE(op) =
    """
        LearnAPI.$(op)_output_type(algorithm)

    Return an upper bound on the type of the output of the `$op` operation.

    # New implementations

    Implementation is optional. The fallback return value is `Any`.

    """

"$(DOC_INPUT_SCITYPE(:predict))"
predict_input_scitype(::Any) = Union{}

"$(DOC_INPUT_OBSERVATION_SCITYPE(:predict))"
predict_input_observation_scitype(::Any) = Union{}

"$(DOC_INPUT_TYPE(:predict))"
predict_input_type(::Any) = Union{}

"$(DOC_INPUT_OBSERVATION_TYPE(:predict))"
predict_input_observation_type(::Any) = Union{}

"$(DOC_INPUT_SCITYPE(:transform))"
transform_input_scitype(::Any) = Union{}

"$(DOC_INPUT_OBSERVATION_SCITYPE(:transform))"
transform_input_observation_scitype(::Any) = Union{}

"$(DOC_INPUT_TYPE(:transform))"
transform_input_type(::Any) = Union{}

"$(DOC_INPUT_OBSERVATION_TYPE(:transform))"
transform_input_observation_type(::Any) = Union{}

"$(DOC_OUTPUT_SCITYPE(:transform))"
transform_output_scitype(::Any) = Any

"$(DOC_OUTPUT_TYPE(:transform))"
transform_output_type(::Any) = Any


# # TWO-ARGUMENT TRAITS

# Here `s` is `:type` or `:scitype`:
const DOC_PREDICT_OUTPUT(s)  =
    """
        LearnAPI.predict_output_$s(algorithm, kind_of_proxy::KindOfProxy)

    Return an upper bound for the $(s)s of predictions of the specified form where
    supported, and otherwise return `Any`. For example, if

        ŷ = predict(model, Distribution(), data)

    successfully returns (i.e., `algorithm` supports predictions of target probability
    distributions) then the following is guaranteed to hold:

        $(s)(ŷ) <: predict_output_$(s)(algorithm, Distribution())

    **Note.** This trait has a single-argument "convenience" version
    `LearnAPI.predict_output_$(s)(algorithm)` derived from this one, which returns a
    dictionary keyed on target proxy types.

    See also [`LearnAPI.KindOfProxy`](@ref), [`predict`](@ref),
    [`predict_input_$(s)`](@ref).

    # New implementations

    Overloading the trait is optional. Here's a sample implementation for a supervised
    regressor type `MyRgs` that only predicts actual values of the target:

    ```julia
    @trait MyRgs predict_output_$(s) = AbstractVector{ScientificTypesBase.Continuous}
    ```

    The fallback method returns `Any`.

    """

"$(DOC_PREDICT_OUTPUT(:scitype))"
predict_output_scitype(algorithm, kind_of_proxy) = Any

"$(DOC_PREDICT_OUTPUT(:type))"
predict_output_type(algorithm, kind_of_proxy) = Any


# # DERIVED TRAITS

name(A) = split(string(constructor(A)), ".") |> last

is_algorithm(A) = !isempty(functions(A))

preferred_kind_of_proxy(algorithm) = first(kinds_of_proxy(algorithm))

const DOC_PREDICT_OUTPUT2(s) =
    """
        LearnAPI.predict_output_$(s)(algorithm)

    Return a dictionary of upper bounds on the $(s) of predictions, keyed on concrete
    subtypes of [`LearnAPI.KindOfProxy`](@ref). Each of these subtypes represents a
    different form of target prediction (`LiteralTarget`, `Distribution`,
    `SurvivalFunction`, etc) possibly supported by `algorithm`, but the existence of a key
    does not guarantee that form is supported.

    As an example, if

        ŷ = predict(model, Distribution(), data...)

    successfully returns (i.e., `algorithm` supports predictions of target probability
    distributions) then the following is guaranteed to hold:

        $(s)(ŷ) <: LearnAPI.predict_output_$(s)s(algorithm)[Distribution]

    See also [`LearnAPI.KindOfProxy`](@ref), [`predict`](@ref),
    [`LearnAPI.predict_input_$(s)`](@ref).

    # New implementations

    This single argument trait should not be overloaded. Instead, overload
    [`LearnAPI.predict_output_$(s)`](@ref)(algorithm, kind_of_proxy).

    """

"$(DOC_PREDICT_OUTPUT2(:scitype))"
predict_output_scitype(algorithm) =
    Dict(T => predict_output_scitype(algorithm, T())
         for T in CONCRETE_TARGET_PROXY_TYPES)

"$(DOC_PREDICT_OUTPUT2(:type))"
predict_output_type(algorithm) =
    Dict(T => predict_output_type(algorithm, T())
         for T in CONCRETE_TARGET_PROXY_TYPES)
