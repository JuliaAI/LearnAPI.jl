const ACCESSOR_FUNCTIONS = (
    :features_importances,
    :training_labels,
    :training_losses,
    :training_scores,
)

"""
    LearnAPI.feature_importances(algorithm, fitted_params, report)

Return the algorithm-specific feature importances of `algorithm`, given `fitted_params` and
`report`, as returned by [`LearnAPI.fit`](@ref), [`LearnAPI.update!`](@ref) or
[`LearnAPI.ingest!`](@ref). The value returned has the form of an abstract vector of
`feature::Symbol => importance::Real` pairs (e.g `[:gender => 0.23, :height => 0.7, :weight
=> 0.1]`).

The `algorithm` supports feature importances if `:feature_importance in
LearnAPI.functions(algorithm)`.

If an algorithm is sometimes unable to report feature importances then
`feature_importances` will return all importances as 0.0, as in `[:gender => 0.0, :height
=> 0.0, :weight => 0.0]`.

# New implementations

`LearnAPI.feature_importances(algorithm::SomeAlgorithmType, fitted_params, report)` may be
overloaded for any type `SomeAlgorithmType` whose instances are algorithms in the LearnAPI
sense. If an algorithm can report multiple feature importance types, then the specific type to
be reported should be controlled by a hyperparameter (i.e., by some property of `algorithm`).

$(DOC_IMPLEMENTED_METHODS(:feature_importances)).

"""
function feature_importances end

"""
    training_losses(algorithm, fitted_params, report)

Return the training losses for `algorithm`, given `fitted_params` and
`report`, as returned by [`LearnAPI.fit`](@ref), [`LearnAPI.update!`](@ref) or
[`LearnAPI.ingest!`](@ref).

# New implementations

Implement for iterative algorithms that compute and record training losses as part of training
(e.g. neural networks).

$(DOC_IMPLEMENTED_METHODS(:training_losses)).

"""
function training_losses end

"""
    training_scores(algorithm, fitted_params, report)

Return the training scores for `algorithm`, given `fitted_params` and
`report`, as returned by [`LearnAPI.fit`](@ref), [`LearnAPI.update!`](@ref) or
[`LearnAPI.ingest!`](@ref).

# New implementations

Implement for algorithms, such as outlier detection algorithms, which associate a score with each
observation during training, where these scores are of interest in later processes (e.g, in
defining normalized scores on new data).

$(DOC_IMPLEMENTED_METHODS(:training_scores)).

"""
function training_scores end

"""
    training_labels(algorithm, fitted_params, report)

Return the training labels for `algorithm`, given `fitted_params` and
`report`, as returned by [`LearnAPI.fit`](@ref), [`LearnAPI.update!`](@ref) or
[`LearnAPI.ingest!`](@ref).

# New implementations

$(DOC_IMPLEMENTED_METHODS(:training_labels)).

"""
function training_labels end
