"""
    LearnAPI.feature_importances(model::M, fitted_params, report)

Given `model` of model type `M` for supporting intrinsic feature importances, return
model-specific feature importances, based on `fittted_params` and `report`, as returned as
parts of the return value of [`LearnAPI.fit`](@ref), [`LearnAPI.update!`](@ref) or
[`LearnAPI.ingest!`](@ref). The value returned has the form of an abstract vector of
`feature::Symbol => importance::Real` pairs (e.g `[:gender =>0.23, :height =>0.7, :weight
=> 0.1]`).

The `model` supports feature importances if `:feature_importance in
LearnAPI.implemented_methods(model)`.

If for some reason a model is sometimes unable to report feature importances, then
`feature_importances` should return all importances as 0.0, as in `[:gender =>0.0, :height
=>0.0, :weight => 0.0]`.

# New model implementations

$(DOC_IMPLEMENTED_METHODS(:feature_importances)).

"""
function feature_importances end
