# There are two types of traits - ordinary traits that an implementation overloads to make
# promises of algorithm behavior, and derived traits, which are never overloaded.

const DOC_UNKNOWN =
    "Returns `\"unknown\"` if the algorithm implementation has "*
    "failed to overload the trait. "
const DOC_ON_TYPE = "The value of the trait must depend only on the type of `algorithm`. "

const DOC_ONLY_ONE =
    "No more than one of the following should be overloaded for an algorithm type: "*
    "`LearnAPI.fit_scitype`, `LearnAPI.fit_type`, `LearnAPI.fit_observation_scitype`, "*
    "`LearnAPI.fit_observation_type`."


const TRAITS = [
    :functions,
    :preferred_kind_of_proxy,
    :position_of_target,
    :position_of_weights,
    :descriptors,
    :is_pure_julia,
    :pkg_name,
    :pkg_license,
    :doc_url,
    :load_path,
    :is_wrapper,
    :human_name,
    :iteration_parameter,
    :fit_keywords,
    :fit_scitype,
    :fit_observation_scitype,
    :fit_type,
    :fit_observation_type,
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

functions() = METHODS = (TRAINING_FUNCTIONS..., OPERATIONS..., ACCESSOR_FUNCTIONS...)
const FUNCTIONS = map(d -> "`:$d`", functions())

"""
    LearnAPI.functions(algorithm)

Return a tuple of symbols, such as `(:fit, :predict)`, corresponding to LearnAPI methods
specifically implemented for objects having the same type as `algorithm`.  If non-empty,
this also guarantees `algorithm` is an algorithm, in the LearnAPI sense. See the Reference
section of the manual for details.

# New implementations

Every LearnAPI method that is not a trait and which is specifically implemented for
`typeof(algorithm)` must be included in the return value of this trait. Specifically, the
return value is a tuple of symbols from this list: $(join(FUNCTIONS, ", ")). To regenerate
this list, do `LearnAPI.functions()`.

See also [`LearnAPI.Algorithm`](@ref).

"""
functions(::Any) = ()


"""
    LearnAPI.preferred_kind_of_proxy(algorithm)

Returns an instance of [`LearnAPI.KindOfProxy`](@ref), unless `LearnAPI.predict` is not
implemented for objects of type `typeof(algorithm)`, in which case it returns `nothing`.

The returned target proxy is generally the one with the smallest computational cost, if
more than one type is supported.

See also [`LearnAPI.predict`](@ref), [`LearnAPI.KindOfProxy`](@ref).

# New implementations

Any algorithm implementing `LearnAPI.predict` must overload this trait.

The trait must return a lone instance `T()` for some concrete subtype `T <:
LearnAPI.KindOfProxy`. List these with `subtypes(LearnAPI.KindOfProxy)` and
`subtypes(LearnAPI.IID)`.

Suppose, for example, we have the following implementation of a supervised learner
returning only probablistic predictions:

```julia
LearnAPI.predict(algorithm::MyNewAlgorithmType, LearnAPI.Distribution(), Xnew) = ...
```

Then we can declare

```julia
@trait MyNewAlgorithmType  preferred_kind_of_proxy = LearnAPI.LiteralTarget()
```

which is shorthand for

```julia
LearnAPI.preferred_kind_of_proxy(::MyNewAlgorithmType) = LearnAPI.Distribution()
```

For more on target variables and target proxies, refer to the LearnAPI documentation.

"""
preferred_kind_of_proxy(::Any) = nothing

"""
    LearnAPI.position_of_target(algorithm)

Return the expected position of the target variable within `data` in calls of the form
[`LearnAPI.fit`](@ref)`(algorithm, verbosity, data...)`.

If this number is `0`, then no target is expected. If this number exceeds `length(data)`,
then `data` is understood to exclude the target variable.

"""
position_of_target(::Any) = 0

"""
    LearnAPI.position_of_weights(algorithm)

Return the expected position of per-observation weights within `data` in
calls of the form [`LearnAPI.fit`](@ref)`(algorithm, verbosity, data...)`.

If this number is `0`, then no weights are expected. If this number exceeds
`length(data)`, then `data` is understood to exclude weights, which are assumed to be
uniform.

"""
position_of_weights(::Any) = 0

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

This trait should return a tuple of symbols, as in `(:classifier, :probabilistic)`.

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

Return a string indicating where the `struct` for `typeof(algorithm)` can be found, beginning
with the name of the package module defining it. For example, a return value of
`"FastTrees.LearnAPI.DecisionTreeClassifier"` means the following julia code will return the
algorithm type:

```julia
import FastTrees
FastTrees.LearnAPI.DecisionTreeClassifier
```

$DOC_UNKNOWN


"""
load_path(::Any) = "unknown"


"""
    LearnAPI.is_wrapper(algorithm)

Returns `true` if one or more properties (fields) of `algorithm` may themselves be
algorithms, and `false` otherwise.

# New implementations

This trait must be overloaded if one or more properties (fields) of `algorithm` may take
algorithm values. Fallback return value is `false`.

$DOC_ON_TYPE


"""
is_wrapper(::Any) = false

"""
    LearnAPI.human_name(algorithm)

A human-readable string representation of `typeof(algorithm)`. Primarily intended for
auto-generation of documentation.

# New implementations

Optional. A fallback takes the type name, inserts spaces and removes capitalization. For
example, `KNNRegressor` becomes `"knn regressor"`. Better would be to overload the trait
to return `"K-nearest neighbors regressor"`. Ideally, this is a "concrete" noun like
`"ridge regressor"` rather than an "abstract" noun like `"ridge regression"`.

"""
human_name(M) = snakecase(name(M), delim=' ') # `name` defined below

"""
    LearnAPI.iteration_parameter(algorithm)

The name of the iteration parameter of `algorithm`, or `nothing` if the algorithm is not
iterative.

# New implementations

Implement if algorithm is iterative. Returns a symbol or `nothing`.

"""
iteration_parameter(::Any) = nothing

"""
    LearnAPI.fit_keywords(algorithm)

Return a list of keywords that can be provided to `fit` that correspond to
metadata; $DOC_METADATA

# New implementations

If `LearnAPI.fit(algorithm, ...)` supports keyword arguments, then this trait must be
overloaded, and otherwise not. Fallback returns `()`.

Here's a sample implementation for a classifier that implements a `LearnAPI.fit` method
with signature `fit(algorithm::MyClassifier, verbosity, X, y; class_weights=nothing)`:

```
LearnAPI.fit_keywords(::Any{<:MyClassifier}) = (:class_weights,)
```

or the shorthand

```
@trait MyClassifier fit_keywords=(:class_weights,)
```


"""
fit_keywords(::Any) = ()

"""
    LearnAPI.fit_scitype(algorithm)

Return an upper bound on the scitype of data guaranteeing it to work when training
`algorithm`.

Specifically, if the return value is `S` and `ScientificTypes.scitype(data) <: S`, then
the following low-level calls are allowed (assuming `metadata` is also valid and
`verbosity` is an integer):

```julia
# apply data front-end:
data2, metadata2 = LearnAPI.reformat(algorithm, LearnAPI.fit, data...; metadata...)

# train:
LearnAPI.fit(algorithm, verbosity, data2...; metadata2...)
```

See also [`LearnAPI.fit_type`](@ref), [`LearnAPI.fit_observation_scitype`](@ref),
[`LearnAPI.fit_observation_type`](@ref).

# New implementations

Optional. The fallback return value is `Union{}`.  $DOC_ONLY_ONE

"""
fit_scitype(::Any) = Union{}

"""
    LearnAPI.fit_observation_scitype(algorithm)

Return an upper bound on the scitype of observations guaranteed to work when training
`algorithm` (independent of the type/scitype of the data container itself).

Specifically, denoting the type returned above by `S`, suppose a user supplies training
data, `data` - typically a tuple, such as `(X, y)` - and valid metadata, `metadata`, and
one computes

    data2, metadata2 = LearnAPI.reformat(algorithm, LearnAPI.fit, data...; metadata...)

Then, assuming

    ScientificTypes.scitype(LearnAPI.getobs(algorithm, LearnAPI.fit, data2, i)) <: S

for any valid index `i`, the following is guaranteed to work:


```julia
LearnAPI.fit(algorithm, verbosity, data2...; metadata2...)
```

See also See also [`LearnAPI.fit_type`](@ref), [`LearnAPI.fit_scitype`](@ref),
[`LearnAPI.fit_observation_type`](@ref).

# New implementations

Optional. The fallback return value is `Union{}`. $DOC_ONLY_ONE

"""
fit_observation_scitype(::Any) = Union{}

"""
    LearnAPI.fit_type(algorithm)

Return an upper bound on the type of data guaranteeing it to work when training `algorithm`.

Specifically, if the return value is `T` and `typeof(data) <: T`, then the following
low-level calls are allowed (assuming `metadata` is also valid and `verbosity` is an
integer):

```julia
# apply data front-end:
data2, metadata2 = LearnAPI.reformat(algorithm, LearnAPI.fit, data...; metadata...)

# train:
LearnAPI.fit(algorithm, verbosity, data2...; metadata2...)
```

See also [`LearnAPI.fit_scitype`](@ref), [`LearnAPI.fit_observation_type`](@ref).
[`LearnAPI.fit_observation_scitype`](@ref)

# New implementations

Optional. The fallback return value is `Union{}`. $DOC_ONLY_ONE

"""
fit_type(::Any) = Union{}

"""
    LearnAPI.fit_observation_type(algorithm)

Return an upper bound on the type of observations guaranteed to work when training
`algorithm` (independent of the type/scitype of the data container itself).

Specifically, denoting the type returned above by `T`, suppose a user supplies training
data, `data` - typically a tuple, such as `(X, y)` - and valid metadata, `metadata`, and
one computes

    data2, metadata2 = LearnAPI.reformat(algorithm, LearnAPI.fit, data...; metadata...)

Then, assuming

    typeof(LearnAPI.getobs(algorithm, LearnAPI.fit, data2, i)) <: T

for any valid index `i`, the following is guaranteed to work:


```julia
LearnAPI.fit(algorithm, verbosity, data2...; metadata2...)
```

See also See also [`LearnAPI.fit_type`](@ref), [`LearnAPI.fit_scitype`](@ref),
[`LearnAPI.fit_observation_scitype`](@ref).

# New implementations

Optional. The fallback return value is `Union{}`. $DOC_ONLY_ONE

"""
fit_observation_type(::Any) = Union{}

DOC_INPUT_SCITYPE(op) =
    """
        LearnAPI.$(op)_input_scitype(algorithm)

    Return an upper bound on the scitype of input data guaranteed to work with the `$op`
    operation.

    Specifically, if `S` is the value returned and `ScientificTypes.scitype(data) <: S`,
    then the following low-level calls are allowed

        data2 = LearnAPI.reformat(algorithm, LearnAPI.$op, data...)
        LearnAPI.$op(algorithm, fitted_params, data2...)

    Here `fitted_params` are the learned parameters returned by an appropriate call to
    `LearnAPI.fit`.

    See also [`LearnAPI.$(op)_input_type`](@ref).

    # New implementations

    Implementation is optional. The fallback return value is `Union{}`. Should not be
    overloaded if `LearnAPI.$(op)_input_type` is overloaded.

   """

DOC_INPUT_TYPE(op) =
    """
        LearnAPI.$(op)_input_type(algorithm)

    Return an upper bound on the type of input data guaranteed to work with the `$op`
    operation.

    Specifically, if `T` is the value returned and `typeof(data) <: S`, then the following
    low-level calls are allowed

        data2 = LearnAPI.reformat(algorithm, LearnAPI.$op, data...)
        LearnAPI.$op(algorithm, fitted_params, data2...)

    Here `fitted_params` are the learned parameters returned by an appropriate call to
    `LearnAPI.fit`.

    See also [`LearnAPI.$(op)_input_scitype`](@ref).

    # New implementations

    Implementation is optional. The fallback return value is `Union{}`. Should not be
    overloaded if `LearnAPI.$(op)_input_scitype` is overloaded.

    """

DOC_OUTPUT_SCITYPE(op) =
    """
        LearnAPI.$(op)_output_scitype(algorithm)

    Return an upper bound on the scitype of the output of the `$op` operation.

    Specifically, if `S` is the value returned, and if

        output, report = LearnAPI.$op(algorithm, fitted_params, data...)

    for suitable `fitted_params` and `data`, then

        ScientificTypes.scitype(output) <: S

    See also [`LearnAPI.$(op)_input_scitype`](@ref).

    # New implementations

    Implementation is optional. The fallback return value is `Any`.

    """

DOC_OUTPUT_TYPE(op) =
    """
        LearnAPI.$(op)_output_type(algorithm)

    Return an upper bound on the type of the output of the `$op` operation.

    Specifically, if `T` is the value returned, and if

        output, report = LearnAPI.$op(algorithm, fitted_params, data...)

    for suitable `fitted_params` and `data`, then

        typeof(output) <: T

    See also [`LearnAPI.$(op)_input_type`](@ref).

    # New implementations

    Implementation is optional. The fallback return value is `Any`.

    """

"$(DOC_INPUT_SCITYPE(:predict))"
predict_input_scitype(::Any) = Union{}

"$(DOC_INPUT_TYPE(:predict))"
predict_input_type(::Any) = Union{}

"$(DOC_INPUT_SCITYPE(:transform))"
transform_input_scitype(::Any) = Union{}

"$(DOC_OUTPUT_SCITYPE(:transform))"
transform_output_scitype(::Any) = Any

"$(DOC_INPUT_TYPE(:transform))"
transform_input_type(::Any) = Union{}

"$(DOC_OUTPUT_TYPE(:transform))"
transform_output_type(::Any) = Any


# # TWO-ARGUMENT TRAITS

# Here `s` is `:type` or `:scitype`:
const DOC_PREDICT_OUTPUT(s)  =
    """
        LearnAPI.predict_output_$s(algorithm, kind_of_proxy::KindOfProxy)

    Return an upper bound for the $(s)s of predictions of the specified form where
    supported, and otherwise return `Any`. For example, if

        ŷ, report = LearnAPI.predict(algorithm, LearnAPI.Distribution(), data...)

    successfully returns (i.e., `algorithm` supports predictions of target probability
    distributions) then the following is guaranteed to hold:

        $(s)(ŷ) <: LearnAPI.predict_output_$(s)(algorithm, LearnAPI.Distribution())

    **Note.** This trait has a single-argument "convenience" version
    `LearnAPI.predict_output_$(s)(algorithm)` derived from this one, which returns a
    dictionary keyed on target proxy types.

    See also [`LearnAPI.KindOfProxy`](@ref), [`LearnAPI.predict`](@ref),
    [`LearnAPI.predict_input_$(s)`](@ref).

    # New implementations

    Overloading the trait is optional. Here's a sample implementation for a supervised
    regressor type `MyRgs` that only predicts actual values of the target:

        LearnAPI.predict(alogrithm::MyRgs, ::LearnAPI.LiteralTarget, data...) = ...
        LearnAPI.predict_output_$(s)(::MyRgs, ::LearnAPI.LiteralTarget) =
            AbstractVector{ScientificTypesBase.Continuous}

    The fallback method returns `Any`.

    """

"$(DOC_PREDICT_OUTPUT(:scitype))"
predict_output_scitype(algorithm, kind_of_proxy) = Any

"$(DOC_PREDICT_OUTPUT(:type))"
predict_output_type(algorithm, kind_of_proxy) = Any


# # DERIVED TRAITS

name(A) = string(typename(A))

is_algorithm(A) = !isempty(functions(A))

const DOC_PREDICT_OUTPUT2(s) =
    """
        LearnAPI.predict_output_$(s)(algorithm)

    Return a dictionary of upper bounds on the $(s) of predictions, keyed on concrete
    subtypes of [`LearnAPI.KindOfProxy`](@ref). Each of these subtypes respresents a
    different form of target prediction (`LiteralTarget`, `Distribution`, `SurvivalFunction`,
    etc) possibly supported by `algorithm`, but the existence of a key does not guarantee
    that form is supported.

    As an example, if

        ŷ, report = LearnAPI.predict(algorithm, LearnAPI.Distribution(), data...)

    successfully returns (i.e., `algorithm` supports predictions of target probability
    distributions) then the following is guaranteed to hold:

        $(s)(ŷ) <: LearnAPI.predict_output_$(s)(algorithm)[LearnAPI.Distribution]

    See also [`LearnAPI.KindOfProxy`](@ref), [`LearnAPI.predict`](@ref),
    [`LearnAPI.predict_input_$(s)`](@ref).

    # New implementations

    This single argument trait should not be overloaded. Instead, overload
    [`LearnAPI.predict_output_$(s)`](@ref)(algorithm, kind_of_proxy). See above.

    """

"$(DOC_PREDICT_OUTPUT2(:scitype))"
predict_output_scitype(algorithm) =
    Dict(T => predict_output_scitype(algorithm, T())
         for T in CONCRETE_TARGET_PROXY_TYPES)

"$(DOC_PREDICT_OUTPUT2(:type))"
predict_output_type(algorithm) =
    Dict(T => predict_output_type(algorithm, T())
         for T in CONCRETE_TARGET_PROXY_TYPES)


