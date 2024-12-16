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

Return the names of features encountered when fitting or updating some `learner` to obtain
`model`.

The value returned value is a vector of symbols.

This method is implemented if `:(LearnAPI.feature_names) in LearnAPI.functions(learner)`.

See also [`fit`](@ref).

# New implementations

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.feature_names)")).

"""
function feature_names end

"""
    LearnAPI.feature_importances(model)

Return the learner-specific feature importances of a `model` output by
[`fit`](@ref)`(learner, ...)` for some `learner`.  The value returned has the form of
an abstract vector of `feature::Symbol => importance::Real` pairs (e.g `[:gender => 0.23,
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

Return a user-friendly tree, in the form of a root object implementing the following
interface defined in AbstractTrees.jl:

- subtypes `AbstractTrees.AbstractNode{T}`
- implements `AbstractTrees.children()`
- implements `AbstractTrees.printnode()`

Such a tree can be visualized using the TreeRecipe.jl package, for example.

See also [`LearnAPI.trees`](@ref).

# New implementations

Implementation is optional.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.tree)")).

"""
function tree end

"""
    LearnAPI.trees(model)

For some ensemble model, return a vector of trees. See [`LearnAPI.tree`](@ref) for the
form of such trees.

See also [`LearnAPI.tree`](@ref).

# New implementations

Implementation is optional.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.trees)")).

"""
function trees end

"""
    LearnAPI.training_losses(model)

Return the training losses obtained when running `model = fit(learner, ...)` for some
`learner`.

See also [`fit`](@ref).

# New implementations

Implement for iterative algorithms that compute and record training losses as part of
training (e.g. neural networks).

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.training_losses)")).

"""
function training_losses end

"""
    LearnAPI.training_predictions(model)

Return internally computed training predictions when running `model = fit(learner, ...)`
for some `learner`.

See also [`fit`](@ref).

# New implementations

Implement for iterative algorithms that compute and record training losses as part of
training (e.g. neural networks).

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.training_predictions)")).

"""
function training_predictions end

"""
    LearnAPI.training_scores(model)

Return the training scores obtained when running `model = fit(learner, ...)` for some
`learner`.

See also [`fit`](@ref).

# New implementations

Implement for learners, such as outlier detection algorithms, which associate a score
with each observation during training, where these scores are of interest in later
processes (e.g, in defining normalized scores for new data).

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.training_scores)")).

"""
function training_scores end

"""
    LearnAPI.components(model)

For a composite `model`, return the component models (`fit` outputs). These will be in the
form of a vector of named pairs, `property_name::Symbol => component_model`. Here
`property_name` is the name of some learner-valued property (hyper-parameter) of
`learner = LearnAPI.learner(model)`.

A composite model is one for which the corresponding `learner` includes one or more
learner-valued properties, and for which `LearnAPI.is_composite(learner)` is `true`.

See also [`is_composite`](@ref).

# New implementations

Implementent if and only if `model` is a composite model.

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.components)")).

"""
function components end

"""
    LearnAPI.training_labels(model)

Return the training labels obtained when running `model = fit(learner, ...)` for some
`learner`.

See also [`fit`](@ref).

# New implementations

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.training_labels)")).

"""
function training_labels end

# :extras intentionally excluded:
const ACCESSOR_FUNCTIONS_WITHOUT_EXTRAS = (
    :(LearnAPI.learner),
    :(LearnAPI.coefficients),
    :(LearnAPI.intercept),
    :(LearnAPI.tree),
    :(LearnAPI.trees),
    :(LearnAPI.feature_names),
    :(LearnAPI.feature_importances),
    :(LearnAPI.training_labels),
    :(LearnAPI.training_losses),
    :(LearnAPI.training_predictions),
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

$(DOC_IMPLEMENTED_METHODS(":(LearnAPI.training_labels)")).

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
