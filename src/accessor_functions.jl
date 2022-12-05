const ACCESSOR_FUNCTIONS = (
    :features_importances,
    :training_labels,
    :training_losses,
    :training_scores,
)

"""
    LearnAPI.feature_importances(model, fitted_params, report)

Return the model-specific feature importances of `model`, given `fittted_params` and
`report`, as returned by [`LearnAPI.fit`](@ref), [`LearnAPI.update!`](@ref) or
[`LearnAPI.ingest!`](@ref). The value returned has the form of an abstract vector of
`feature::Symbol => importance::Real` pairs (e.g `[:gender =>0.23, :height =>0.7, :weight
=> 0.1]`).

The `model` supports feature importances if `:feature_importance in
LearnAPI.functions(model)`.

If for some reason a model is sometimes unable to report feature importances, then
`feature_importances` will return all importances as 0.0, as in `[:gender =>0.0, :height
=>0.0, :weight => 0.0]`.

# New model implementations

`LearnAPI.feature_importances(model::SomeModelType, fitted_params, report)` may be
overloaded for any type `SomeModelType` whose instances are models in the LearnAPI
sense. If a model can report multiple feature importance types, then the specific type to
be reported should be controlled by a hyperparameter (i.e., by some property of `model`).

$(DOC_IMPLEMENTED_METHODS(:feature_importances)).

"""
function feature_importances end

function training_labels end
function training_losses end
function training_scores end
