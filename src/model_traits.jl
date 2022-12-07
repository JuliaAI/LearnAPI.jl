# There are two types of traits - ordinary traits that an implemenation overloads to make
# promises of model behaviour, and derived traits, which are never overloaded.

const DERIVED_TRAITS = (:name, :ismodel)
const ORDINARY_TRAITS = (
    :functions,
    :target_proxies,
    :position_of_target,
    :position_of_weights,
    :descriptors,
    :is_pure_julia,
    :pkg_name,
    :pkg_license,
    :doc_url,
    :load_path,
    :is_wrapper,
    :fit_keywords,
    :human_name,
    :iteration_parameter,
    :fit_data_scitype,
    :fit_data_type,
    :fit_observation_scitypes,
    :fit_observation_types,
    :input_scitypes,
    :input_types,
    :output_scitypes,
    :output_types,
)

# # DERIVED TRAITS

name(M::Type) = string(typename(M))
ismodel(M::Type) = !isempty(functions(M))


# # ORDINARY TRAITS

functions() = METHODS = (OPERATIONS..., ACCESSOR_FUNCTIONS...)

"""
   LearnAPI.functions(m)

Return a tuple of symbols, such as `(:fit, :predict)`, corresponding to LearnAPI.jl
methods implemented for objects having the same type as `m`.

If non-empty, this also guarantees `m` is a *model*, in the LearnAPI.jl
sense.

Specifically, this means:

$DOC_MODEL

# New model implementations

Elements of the returned tuple must come from this list: $(join(string.(functions()), ", ")).

A new type whose instances are intended to be LearnAPI models can guarantee that by
subtyping [`LearnAPI.Model`](@ref).

See also [`LearnAPI.Model`](@ref).

"""
functions(::Type) = ()

target_proxies() = subtypes(TargetProxy)

"""
    target_proxies(model)

Return a named tuple of target proxies, keyed on operation name, applying to `model`. For
example, a value of

    (predict=LearnAPI.Distribution(),)

means that `LearnAPI.predict` returns probability distributions, rather than actual values
of the target. View all target proxy types with `target_proxies()`. For more information
on target variables and target proxies, refer to the LearnAPI documentation.

"""
target_proxies(::Type) = NamedTuple()

position_of_target(::Type) = 0

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
    :natural_language,
    :image_processing,
    :audio_processing,
]
descriptors(::Type) = ()

is_pure_julia(::Type) = false

pkg_name(::Type) = "Unknown"

doc_url(::Type) = "unknown"

pkg_license(::Type) = "unknown"

load_path(::Type) = "unknown"

is_wrapper(::Type) = false

fit_keywords(::Type) = ()

human_name(M::Type{}) = snakecase(name(M), delim=' ') # `name` defined below

iteration_parameter(::Type) = nothing

fit_data_scitype(::Type) = Union{}

fit_data_type(::Type) = Union{}

fit_observation_scitypes(::Type) = Union{}

fit_observation_types(::Type) = Union{}

input_scitypes(::Type) = NamedTuple()

input_types(::Type) = NamedTuple()

output_scitypes(::Type) = NamedTuple()

output_types(::Type) = NamedTuple()
