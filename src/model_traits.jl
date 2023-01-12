# There are two types of traits - ordinary traits that an implemenation overloads to make
# promises of model behaviour, and derived traits, which are never overloaded.

const DOC_UNKNOWN =
    "Returns \"unknown\" if the model implementation has failed to overload the trait. "
const DOC_ON_TYPE = "The value of the trait must depend only on the type of `model`. "

const ORDINARY_TRAITS = (
    :functions,
    :predict_proxy,
    :predict_joint_proxy,
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
    :predict_joint_input_scitype,
    :predict_joint_output_scitype,
    :predict_joint_input_type,
    :predict_joint_output_type,
    :transform_input_scitype,
    :transform_output_scitype,
    :transform_input_type,
    :transform_output_type,
    :inverse_transform_input_scitype,
    :inverse_transform_output_scitype,
    :inverse_transform_input_type,
    :inverse_transform_output_type,
)
const DERIVED_TRAITS = (:name, :is_model)

# # ORDINARY TRAITS

functions() = METHODS = (TRAINING_FUNCTIONS..., OPERATIONS..., ACCESSOR_FUNCTIONS...)
const FUNCTIONS = map(d -> "`:$d`", functions())

"""
    LearnAPI.functions(model)

Return a tuple of symbols, such as `(:fit, :predict)`, corresponding to LearnAPI
methods specifically implemented for objects having the same type as `model`.

If non-empty, this also guarantees `model` is a model, in the LearnAPI sense. See the
Reference section of the manual for details.

# New model implementations

Every LearnAPI method that is not a trait and which is specifically implemented for
`typeof(model)` must be included in the return value of this trait. Specifically, the
return value is a tuple of symbols from this list: $(join(FUNCTIONS, ", ")). To regenerate
this list, do `LearnAPI.functions()`.

See also [`LearnAPI.Model`](@ref).

"""
functions(::Type) = ()


"""
    LearnAPI.predict_proxy(model)

Returns an object with abstract type `LearnAPI.TargetProxy` indicating the kind of proxy
for the target returned by the `predict` method, when called on `model` and some data. For
example, a value of `LearnAPI.Distribution()` means that `predict` returns probability
distributions, rather than actual values of the target. (`LearnAPI.predict` also retuns a
report as second value). A value of `LearnAPI.TrueTarget()` indicates that ordinary
(non-proxy) target values are returned.

# New implementations

For more on target variables and target proxies, refer to the "Predict and Other
Operations" section of the LearnAPI documentation.

A model with a concept of "target" must overload this trait.

The trait must return a lone instance `T()` for some subtype `T <: LearnAPI.TargetProxy`.
Here's a sample implementation for a supervised model where predictions are ordinary
values of the target variable:

```julia
@trait MyNewModel predict_proxy = LearnAPI.TrueTarget()
```

which is shorthand for

```julia
LearnAPI.predict_proxy(::Type{<:MyNewModelType}) = LearnAPI.TrueTarget()
```

"""
predict_proxy(::Type) = NamedTuple()

"""
    LearnAPI.predict_joint_proxy(model)

Returns an object with abstract type `LearnAPI.TargetProxy` indicating the kind of proxy
for the target returned by the `predict_joint` method, when called on `model` and some
data. For example, a value of `LearnAPI.Distribution()` means that `predict_joint` returns
a probability distribution, rather than, say a merely sampleable object.

# New implementations

For more on target variables and target proxies, refer to the LearnAPI documentation.

Any model implementing `LearnAPI.predict_joint` must overload this trait.

The possible return values for this trait are: `LearnAPI.JointSampleable()`,
`LearnAPI.JointDistribution()` and `LearnAPI.JointLogDistribution()`.

Here's a sample implementation:

```julia
@trait MyNewModel predict_joint_proxy = LearnAPI.JointDistribution()
```

which is shorthand for

```julia
LearnAPI.predict_joint_proxy(::Type{<:MyNewModelType}) = LearnAPI.JointDistribution()
```

"""
predict_joint_proxy(::Type) = NamedTuple()

"""
    LearnAPI.position_of_target(model)

Return the expected position of the target variable within `data` in calls of the form
[`LearnAPI.fit`](@ref)(model, verbosity, data...)`.

If this number is `0`, then no target is expected. If this number exceeds `length(data)`,
then `data` is understood to exclude the target variable.

"""
position_of_target(::Type) = 0

"""
    LearnAPI.position_of_weights(model)

Return the expected position of per-observation weights within `data` in
calls of the form [`LearnAPI.fit`](@ref)(model, verbosity, data...)`.

If this number is `0`, then no weights are expected. If this number exceeds
`length(data)`, then `data` is understood to exclude weights, which are assumed to be
uniform.

"""
position_of_weights(::Type) = 0

descriptors() = [
    :regression,
    :classification,
    :clustering,
    :gradient_descent,
    :iterative_model,
    :incremental_model,
    :dimension_reduction,
    :transformer,
    :static_transformer,
    :missing_value_imputer,
    :ensemble_model,
    :wrapper,
    :time_series_forecaster,
    :time_series_classifier,
    :survival_analysis,
    :distribution_fitter,
    :Bayesian_model,
    :outlier_detection,
    :collaborative_filtering,
    :text_analysis,
    :audio_analysis,
    :natural_language_processing,
    :image_processing,
]

const DESCRIPTORS = map(d -> "`:$d`", descriptors())

"""
    LearnAPI.descriptors(model)

Lists one or more suggestive model descriptors from this list: $(join(DESCRIPTORS, ", ")).

!!! warning
    The value of this trait guarantees no particular behaviour. The trait is
    intended for informal classification purposes only.

# New model implementations

This trait should return a tuple of symbols, as in `(:classifier, :probabilistic)`.

"""
descriptors(::Type) = ()

"""
    LearnAPI.is_pure_julia(model)

Returns `true` if training `model` requires evaluation of pure Julia code only.

# New model implementations

The fallback is `false`.

"""
is_pure_julia(::Type) = false

"""
    LearnAPI.pkg_name(model)

Return the name of the package module which supplies the core training algorithm for
`model`.  This is not necessarily the package providing the LearnAPI
interface.

$DOC_UNKNOWN

# New model implemetations

Must return a string, as in "DecisionTree".

"""
pkg_name(::Type) = "Unknown"

"""
    LearnAPI.doc_url(model)

Return a url where the core algorithm for `model` is documented.

$DOC_UNKNOWN

# New model implementations

Must return a string, such as "https://en.wikipedia.org/wiki/Decision_tree_learning".

"""
doc_url(::Type) = "unknown"

"""
    LearnAPI.pkg_license(model)

Return the name of the software license, such as "MIT", applying to the package where the
core algorithm for `model` is implemented.

"""
pkg_license(::Type) = "unknown"

"""
    LearnAPI.load_path(model)

Return a string indicating where the `struct` for `typeof(model)` can be found, beginning
with the name of the package module defining it. For example, a return value of
"FastTrees.LearnAPI.DecisionTreeClassifier" means the following julia code will return the
model type:

```julia
import FastTrees
FastTrees.LearnAPI.DecisionTreeClassifier
```

$DOC_UNKNOWN


"""
load_path(::Type) = "unknown"


"""
    LearnAPI.is_wrapper(model)

Returns `true` if one or more properties (fields) of `model` are themselves models, and
`false` otherwise.

# New model implementations

This trait must be overloaded if one or more properties (fields) of `model` are themselves
models. Fallback return value is `false`. 

$DOC_ON_TYPE


"""
is_wrapper(::Type) = false

"""
    LearnAPI.human_name(model)

A human-readable string representation of `typeof(model)`. Primarily intended for
auto-generation of documentation.

# New model implementations

Optional. A fallback takes the type name, inserts spaces and removes capitalization. For
example, `KNNRegressor` becomes "knn regressor". Better would be to overload the trait to
give "K-nearest neighbors regressor". Should be "concrete" noun like "ridge regressor"
rather than an "abstract" noun like "ridge regression".

"""
human_name(M::Type{}) = snakecase(name(M), delim=' ') # `name` defined below

"""
    LearnAPI.iteration_parameter(model)

The name of the iteration parameter of `model`, or `nothing` if the model is not
iterative.

# New model implementations

Implement if model is iterative. Returns a symbol or nothing.

"""
iteration_parameter(::Type) = nothing

"""
    LearnAPI.fit_keywords(model)

Return a list of keywords that can be provided to `fit` that correspond to
metadata. $DOC_METADATA

# New model implementations

If `LearnAPI.fit(model, ...)` supports keyword arguments, then this trait must be
overloaded, and otherwise not. Fallback returns `()`.

"""
fit_keywords(::Type) = ()

"""
    LearnAPI.fit_scitype(model)

Return an upper bound on the scitype of data guaranteeing it to work when training
`model`.

Specifically, if the return value is `S` and `ScientificTypes.scitype(data) <: S`, then
the following low-level calls are allowed (assuming `metadata` is also valid and
`verbosity` is an integer):

```julia
# apply data front-end:
data2, metadata2 = LearnAPI.reformat(model, LearnAPI.fit, data...; metadata...)

# train:
LearnAPI.fit(model, verbosity, data2...; metadata2...)
```

# New model implementations

Optional. The fallback return value is `Union{}`.

See also [`LearnAPI.fit_type`](@ref), [`LearnAPI.fit_observation_scitype`](@ref),
[`LearnAPI.fit_observation_type`](@ref).

"""
fit_scitype(::Type) = Union{}

"""
    LearnAPI.fit_observation_scitype(model)

Return an upper bound on the scitype of observations guaranteed to work when training
`model` (independent of the type/scitype of the data container itself).

Specifically, denoting the type returned above by `S`, suppose a user supplies training
data, `data` - typically a tuple, such as `(X, y)` - and valid metadata, `metadata`, and
one computes

    data2, metadata2 = LearnAPI.reformat(model, LearnAPI.fit, data...; metadata...)

Then, assuming

    ScientificTypes.scitype(LearnAPI.getobs(model, LearnAPI.fit, data2, i)) <: S

for any valid index `i`, the following is guaranteed to work:


```julia
LearnAPI.fit(model, verbosity, data2...; metadata2...)
```

# New model implementations

Optional. The fallback return value is `Union{}`.

See also See also [`LearnAPI.fit_type`](@ref), [`LearnAPI.fit_scitype`](@ref),
[`LearnAPI.fit_observation_type`](@ref).

"""
fit_observation_scitype(::Type) = Union{}

"""
    LearnAPI.fit_type(model)

Return an upper bound on the type of data guaranteeing it to work when training `model`.

Specifically, if the return value is `T` and `typeof(data) <: T`, then the following
low-level calls are allowed (assuming `metadata` is also valid and `verbosity` is an
integer):

```julia
# apply data front-end:
data2, metadata2 = LearnAPI.reformat(model, LearnAPI.fit, data...; metadata...)

# train:
LearnAPI.fit(model, verbosity, data2...; metadata2...)
```

# New model implementations

Optional. The fallback return value is `Union{}`.

See also [`LearnAPI.fit_scitype`](@ref)

"""
fit_type(::Type) = Union{}

"""
    LearnAPI.fit_observation_type(model)

Return an upper bound on the type of observations guaranteed to work when training
`model` (independent of the type/scitype of the data container itself).

Specifically, denoting the type retuned above by `T`, suppose a user supplies training
data, `data` - typically a tuple, such as `(X, y)` - and valid metadata, `metadata`, and
one computes

    data2, metadata2 = LearnAPI.reformat(model, LearnAPI.fit, data...; metadata...)

Then, assuming

    typeof(LearnAPI.getobs(model, LearnAPI.fit, data2, i)) <: T

for any valid index `i`, the following is guaranteed to work:


```julia
LearnAPI.fit(model, verbosity, data2...; metadata2...)
```

# New model implementations

Optional. The fallback return value is `Union{}`.

See also See also [`LearnAPI.fit_type`](@ref), [`LearnAPI.fit_scitype`](@ref),
[`LearnAPI.fit_observation_scitype`](@ref).

"""
fit_observation_type(::Type) = Union{}

DOC_INPUT_SCITYPE(op) =
    """
        $(op)_input_scitype(model)

    Return an upper bound on the scitype of input data guaranteed to work with the `$op`
    operation.

    Specifically, if `S` is the value returned and `ScientificTypes.scitype(data) <: S`,
    then the following low-level calls are allowed

        data2 = LearnAPI.reformat(model, LearnAPI.$op, data...)
        LearnAPI.$op(model, fitted_params, data2...)

    Here `fitted_params` are the learned parameters returned by an appropriate call to
    `LearnAPI.fit`.

    See also [`$(op)_input_type`](@ref).

    """

DOC_INPUT_TYPE(op) =
    """
        $(op)_input_type(model)

    Return an upper bound on the type of input data guaranteed to work with the `$op`
    operation.

    Specifically, if `T` is the value returned and `typeof(data) <: S`, then the following
    low-level calls are allowed

        data2 = LearnAPI.reformat(model, LearnAPI.$op, data...)
        LearnAPI.$op(model, fitted_params, data2...)

    Here `fitted_params` are the learned parameters returned by an appropriate call to
    `LearnAPI.fit`.

    See also [`$(op)_input_scitype`](@ref).

    """

DOC_OUTPUT_SCITYPE(op) =
    """
        $(op)_output_scitype(model)

    Return an upper bound on the scitype of the output of the `$op` operation.

    Specifically, if `S` is the value returned, and if

        output, report = LearnAPI.$op(model, fitted_params, data...)

    for suitable `fitted_params` and `data`, then

        ScientificTypes.scitype(output) <: S

    See also [`$(op)_input_scitype`](@ref).

    """

DOC_OUTPUT_TYPE(op) =
    """
        $(op)_output_type(model)

    Return an upper bound on the type of the output of the `$op` operation.

    Specifically, if `T` is the value returned, and if

        output, report = LearnAPI.$op(model, fitted_params, data...)

    for suitable `fitted_params` and `data`, then

        typeof(output) <: T

    See also [`$(op)_input_type`](@ref).

    """

"$(DOC_INPUT_SCITYPE(:predict))"
predict_input_scitype(::Type) = Union{}

"$(DOC_OUTPUT_SCITYPE(:predict))"
predict_output_scitype(::Type) = Any

"$(DOC_INPUT_TYPE(:predict))"
predict_input_type(::Type) = Union{}

"$(DOC_OUTPUT_TYPE(:predict))"
predict_output_type(::Type) = Any

"$(DOC_INPUT_SCITYPE(:predict_joint))"
predict_joint_input_scitype(::Type) = Union{}

"$(DOC_OUTPUT_SCITYPE(:predict_joint))"
predict_joint_output_scitype(::Type) = Any

"$(DOC_INPUT_TYPE(:predict_joint))"
predict_joint_input_type(::Type) = Union{}

"$(DOC_OUTPUT_TYPE(:predict_joint))"
predict_joint_output_type(::Type) = Any

"$(DOC_INPUT_SCITYPE(:transform))"
transform_input_scitype(::Type) = Union{}

"$(DOC_OUTPUT_SCITYPE(:transform))"
transform_output_scitype(::Type) = Any

"$(DOC_INPUT_TYPE(:transform))"
transform_input_type(::Type) = Union{}

"$(DOC_OUTPUT_TYPE(:transform))"
transform_output_type(::Type) = Any

"$(DOC_INPUT_SCITYPE(:inverse_transform))"
inverse_transform_input_scitype(::Type) = Union{}

"$(DOC_OUTPUT_SCITYPE(:inverse_transform))"
inverse_transform_output_scitype(::Type) = Any

"$(DOC_INPUT_TYPE(:inverse_transform))"
inverse_transform_input_type(::Type) = Union{}

"$(DOC_OUTPUT_TYPE(:inverse_transform))"
inverse_transform_output_type(::Type) = Any


# # DERIVED TRAITS

name(M::Type) = string(typename(M))
is_model(M::Type) = !isempty(functions(M))
