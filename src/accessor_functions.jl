# # NOTE ON ADDING NEW ACCESSOR FUNCTIONS

# - Add new accessor function to ACCESSOR_FUNCTIONS_WITHOUT_EXTRAS,
#   defined near the end of this file.

# - Update the documentation page /docs/src/accesssor_functions.md


const DOC_STATIC =
    """

    For "static" learners (those without training `data`) it may be necessary to first
    call `transform` or `predict` on `model`.

    """

"""
    LearnAPI.learner(model)
    LearnAPI.learner(stripped_model)

Recover the learner used to train `model` or the output, `stripped_model`, of
[`LearnAPI.strip(model)`](@ref).

In other words, if `model = fit(learner, data...)`, for some `learner` and `data`,
then

```julia
LearnAPI.learner(model) == learner == LearnAPI.learner(LearnAPI.strip(model))
```
is `true`.

# New implementations

Implementation is compulsory for new learner types. The behaviour described above is the
only contract. You must include `:(LearnAPI.learner)` in the return value of
[`LearnAPI.functions(learner)`](@ref).

"""
function learner end

"""
    LearnAPI.strip(model; options...)

Return a version of `model` that will generally have a smaller memory allocation than
`model`, suitable for serialization. Here `model` is any object returned by
[`fit`](@ref). Accessor functions that can be called on `model` may not work on
`LearnAPI.strip(model)`, but [`predict`](@ref), [`transform`](@ref) and
[`inverse_transform`](@ref) will work, if implemented. Check
`LearnAPI.functions(LearnAPI.learner(model))` to view see what the original `model`
implements.

Implementations may provide learner-specific keyword `options` to control how much of the
original functionality is preserved by `LearnAPI.strip`.

# Typical workflow

```julia
model = fit(learner, (X, y)) # or `fit(learner, X, y)`
ŷ = predict(model, Point(), Xnew)

small_model = LearnAPI.strip(model)
serialize("my_model.jls", small_model)

recovered_model = deserialize("my_random_forest.jls")
@assert predict(recovered_model, Point(), Xnew) == ŷ
```

# Extended help

# New implementations

Overloading `LearnAPI.strip` for new learners is optional. The fallback is the
identity.

New implementations must enforce the following identities, whenever the right-hand side is
defined:

```julia
predict(LearnAPI.strip(model; options...), args...; kwargs...) ==
    predict(model, args...; kwargs...)
transform(LearnAPI.strip(model; options...), args...; kwargs...) ==
    transform(model, args...; kwargs...)
inverse_transform(LearnAPI.strip(model; options), args...; kwargs...) ==
    inverse_transform(model, args...; kwargs...)
```

Additionally:

```julia
LearnAPI.strip(LearnAPI.strip(model)) == LearnAPI.strip(model)
```

"""
LearnAPI.strip(model) = model

"""
    LearnAPI.feature_names(model)

Where supported, return the names of features encountered when fitting or updating some
`learner` to obtain `model`.

The value returned value is a vector of symbols.

This method is implemented if `:(LearnAPI.feature_names) in LearnAPI.functions(learner)`.

See also [`fit`](@ref).

# New implementations

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.feature_names)")).

"""
function feature_names end

"""
    LearnAPI.feature_importances(model)

Where supported, return the learner-specific feature importances of a `model` output by
[`fit`](@ref)`(learner, ...)` for some `learner`.  The value returned has the form of an
abstract vector of `feature::Symbol => importance::Real` pairs (e.g `[:gender => 0.23,
:height => 0.7, :weight => 0.1]`).

The `learner` supports feature importances if `:(LearnAPI.feature_importances) in
LearnAPI.functions(learner)`.

If a learner is sometimes unable to report feature importances then
`LearnAPI.feature_importances` will return all importances as 0.0, as in `[:gender => 0.0,
:height => 0.0, :weight => 0.0]`.

# New implementations

Implementation is optional.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.feature_importances)")).

"""
function feature_importances end

"""
    LearnAPI.coefficients(model)

For a linear model, return the learned coefficients.  The value returned has the form of
an abstract vector of `feature_or_class::Symbol => coefficient::Real` pairs (e.g `[:gender
=> 0.23, :height => 0.7, :weight => 0.1]`) or, in the case of multi-targets,
`feature::Symbol => coefficients::AbstractVector{<:Real}` pairs.

The `model` reports coefficients if `:(LearnAPI.coefficients) in
LearnAPI.functions(Learn.learner(model))`.

See also [`LearnAPI.intercept`](@ref).

# New implementations

Implementation is optional.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.coefficients)")).

"""
function coefficients end

"""
    LearnAPI.intercept(model)

For a linear model, return the learned intercept.  The value returned is `Real` (single
target) or an `AbstractVector{<:Real}` (multi-target).

The `model` reports intercept if `:(LearnAPI.intercept) in
LearnAPI.functions(Learn.learner(model))`.

See also [`LearnAPI.coefficients`](@ref).

# New implementations

Implementation is optional.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.intercept)")).

"""
function intercept end

"""
    LearnAPI.tree(model)

Return a user-friendly `tree`, implementing the AbstractTrees.jl interface. In particular,
such a tree can be visualized using `AbstractTrees.print_tree(tree)` or using the
TreeRecipe.jl package.

See also [`LearnAPI.trees`](@ref).

# New implementations

Implementation is optional. The returned object should implement the following interface
defined in AbstractTrees.jl:

- `tree` subtypes `AbstractTrees.AbstractNode{T}`

- `AbstractTrees.children(tree)`

- `AbstractTrees.printnode(tree)` should be human-readable

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.tree)")).

"""
function tree end

"""
    LearnAPI.trees(model)

For tree ensemble model, return a vector of trees, each implementing the AbstractTrees.jl
interface.

See also [`LearnAPI.tree`](@ref).

# New implementations

Implementation is optional. See [`LearnAPI.tree`](@ref) for the interface each tree in the
ensemble should implement.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.trees)")).

"""
function trees end

"""
    LearnAPI.training_losses(model)

Return internally computed training losses obtained when running `model = fit(learner,
...)` for some `learner`, one for each iteration of the algorithm. This will be a
numerical vector. The metric used to compute the loss is generally learner-specific, but
may be a user-specifiable learner hyperparameter. Generally, the smaller the loss, the
better the performance.

See also [`fit`](@ref).

# New implementations

Implement for iterative algorithms that compute measures of training performance as part
of training (e.g. neural networks). Return one value per iteration, in chronological
order, with an optional pre-training initial value. If scores are being computed rather
than losses, ensure values are multiplied by -1.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.training_losses)")).

"""
function training_losses end

"""
    LearnAPI.out_of_sample_losses(model)

Where supported, return internally computed out-of-sample losses obtained when running
`model = fit(learner, ...)` for some `learner`, one for each iteration of the
algorithm. This will be a numeric vector. The metric used to compute the loss is generally
learner-specific, but may be a user-specifiable learner hyperparameter. Generally, the
smaller the loss, the better the performance.

If the learner is not setting aside a separate validation set, then the losses are all
`Inf`.

See also [`fit`](@ref).

# New implementations

Only implement this method for learners that specifically allow for the supplied training
data to be internally split into separate "train" and "validation" subsets, and which
additionally compute an out-of-sample loss.  Return one value per iteration, in
chronological order, with an optional pre-training initial value. If scores are being
computed rather than losses, ensure values are multiplied by -1.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.out_of_sample_losses)")).

"""
function out_of_sample_losses end

"""
    LearnAPI.predictions(model)

Where supported, return internally computed predictions on the training `data` after
running `model = fit(learner, data)` for some `learner`. Semantically equivalent to calling
`LearnAPI.predict(model, X)`, where `X = LearnAPI.features(obs(learner, data))` but
generally cheaper.

See also [`fit`](@ref).

# New implementations

Implement for algorithms that internally compute predictions for the training
data. Predictions for the complete test data must be returned, even if only a subset is
internally used for training. Cannot be implemented for static algorithms (algorithms for
which `fit` consumes no data). Here are some possible use cases:

- Clustering algorithms that generalize to new data, but by first learning labels for the
  training data (e.g., K-means); use `predictions(model)` to expose these labels
  to the user so they can avoid the expense of a separate `predict` call.

- Iterative learners such as neural networks, that need to make in-sample predictions
   to estimate to estimate an in-sample loss; use `predictions(model)`
   to expose these predictions to the user so they can avoid a separate `predict` call.

- Ensemble learners, such as gradient tree boosting algorithms, may split the training
  data into internal train and validation subsets and can efficiently build up predictions
  on both with an update for each new ensemble member; expose these predictions to the
  user (for external iteration control, for example) using `predictions(model)` and
  articulate the actual split used using [`LearnAPI.out_of_sample_indices(model)`](@ref).

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.predictions)")).

"""
function predictions end

"""
    LearnAPI.out_of_sample_indices(model)

For a learner also implementing [`LearnAPI.predictions`](@ref), return a vector of
observation indices identifying which part, if any, of `yhat =
LearnAPI.predictions(model)`, is actually out-of-sample predictions. If the learner
trained on all data this will be an empty vector.

Here's a sample workflow for some such `learner`, with training data, `(X, y)`, where `y`
is the training target, here assumed to be a vector.

```julia
import MLUtils.getobs
model = fit(learner, (X, y))
yhat = LearnAPI.predictions(model)
test_indices = LearnAPI.out_of_sample_indices(model)
out_of_sample_loss = yhat[test_indices] .!= y[test_indices] |> mean
```

# New implementations

Implement for algorithms that internally split training data into "train" and
"validate" subsets. Assumes
[`LearnAPI.data_interface(learner)`](@ref)`==LearnAPI.RandomAccess()`.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.out_of_sample_indices)")).
"""
function out_of_sample_indices end

"""
    LearnAPI.training_scores(model)

Where supported, return the training scores obtained when running `model = fit(learner,
...)` for some `learner`. This will be a numerical vector whose length coincides with the
number of training observations, and whose interpretation depends on the learner.

See also [`fit`](@ref).

# New implementations

Implement for learners, such as outlier detection algorithms, which associate a numerical
score with each observation during training, when these scores are of interest in
workflows (e.g, to normalize the scores for new observations).

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.training_scores)")).

"""
function training_scores end

"""
    LearnAPI.components(model)

For a composite `model`, return the component models (`fit` outputs). These will be in the
form of a vector of named pairs, `sublearner::Symbol => component_model(s)`, one for each
`sublearner` in [`LearnAPI.learners(learner)`](@ref), where `learner =
LearnAPI.learner(model)`. Here `component_model(s)` will be the `fit` output (or vector of
`fit` outputs) generated internally for the the corresponding sublearner.

The `model` is composite if [`LearnAPI.learners(learner)`](@ref) is non-empty.

See also [`LearnAPI.learners`](@ref).

# New implementations

Implementent if and only if `model` is a composite model.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.components)")).

"""
function components end

# :extras intentionally excluded:
const ACCESSOR_FUNCTIONS_WITHOUT_EXTRAS = (
    :(LearnAPI.learner),
    :(LearnAPI.coefficients),
    :(LearnAPI.intercept),
    :(LearnAPI.tree),
    :(LearnAPI.trees),
    :(LearnAPI.feature_names),
    :(LearnAPI.feature_importances),
    :(LearnAPI.training_losses),
    :(LearnAPI.out_of_sample_losses),
    :(LearnAPI.predictions),
    :(LearnAPI.out_of_sample_indices),
    :(LearnAPI.training_scores),
    :(LearnAPI.components),
)

const ACCESSOR_FUNCTIONS_WITHOUT_EXTRAS_LIST = join(
    map(ACCESSOR_FUNCTIONS_WITHOUT_EXTRAS) do f
    "[`$f`](@ref)"
    end,
    ", ",
    " and ",
)

 """
    LearnAPI.extras(model)

Return miscellaneous byproducts of a learning algorithm's execution, from the
object `model` returned by a call of the form `fit(learner, data)`.

$DOC_STATIC

See also [`fit`](@ref).

# New implementations

Implementation is discouraged for byproducts already covered by other LearnAPI.jl accessor
functions: $ACCESSOR_FUNCTIONS_WITHOUT_EXTRAS_LIST.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.extras)")).

"""
function extras end

const ACCESSOR_FUNCTIONS =
    (:(LearnAPI.extras), ACCESSOR_FUNCTIONS_WITHOUT_EXTRAS...)

const ACCESSOR_FUNCTIONS_LIST = join(
    map(ACCESSOR_FUNCTIONS) do f
    "[`$f`](@ref)"
    end,
    ", ",
    " and ",
)
