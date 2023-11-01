# # NOTE ON ADDING NEW ACCESSOR FUNCTIONS

# - Add new accessor function to ACCESSOR_FUNCTIONS_WITHOUT_EXTRAS,
#   defined near the end of this file.

# - Update the documentation page /docs/src/accesssor_functions.md


const DOC_STATIC =
    """

    For "static" algorithms (those without training `data`) it may be necessary to first
    call `transform` or `predict` on `model`.

    """

"""
    LearnAPI.algorithm(model)
    LearnAPI.algorithm(minimized_model)

Recover the algorithm used to train `model` or the output of [`minimize(model)`](@ref).

In other words, if `model = fit(algorithm, data...)`, for some `algorithm` and `data`,
then

```julia
LearnAPI.algorithm(model) == algorithm == LearnAPI.algorithm(minimize(model))
```
is `true`.

# New implementations

Implementation is compulsory for new algorithm types. The behaviour described above is the
only contract. $(DOC_IMPLEMENTED_METHODS(:algorithm))

"""
function algorithm end

"""
    LearnAPI.feature_importances(model)

Return the algorithm-specific feature importances of a `model` output by
[`fit`](@ref)`(algorithm, ...)` for some `algorithm`.  The value returned has the form of
an abstract vector of `feature::Symbol => importance::Real` pairs (e.g `[:gender => 0.23,
:height => 0.7, :weight => 0.1]`).

The `algorithm` supports feature importances if `LearnAPI.feature_importances in
LearnAPI.functions(algorithm)`.

If an algorithm is sometimes unable to report feature importances then
`LearnAPI.feature_importances` will return all importances as 0.0, as in `[:gender => 0.0,
:height => 0.0, :weight => 0.0]`.

# New implementations

Implementation is optional.

$(DOC_IMPLEMENTED_METHODS(:feature_importances)).

"""
function feature_importances end

"""
    LearnAPI.coefficients(model)

For a linear model, return the learned coefficients.  The value returned has the form of
an abstract vector of `feature_or_class::Symbol => coefficient::Real` pairs (e.g `[:gender
=> 0.23, :height => 0.7, :weight => 0.1]`) or, in the case of multi-targets,
`feature::Symbol => coefficients::AbstractVector{<:Real}` pairs.

The `model` reports coefficients if `LearnAPI.coefficients in
LearnAPI.functions(Learn.algorithm(model))`.

See also [`LearnAPI.intercept`](@ref).

# New implementations

Implementation is optional.

$(DOC_IMPLEMENTED_METHODS(:coefficients)).

"""
function coefficients end

"""
    LearnAPI.intercept(model)

For a linear model, return the learned intercept.  The value returned is `Real` (single
target) or an `AbstractVector{<:Real}` (multi-target).

The `model` reports intercept if `LearnAPI.intercept in
LearnAPI.functions(Learn.algorithm(model))`.

See also [`LearnAPI.coefficients`](@ref).

# New implementations

Implementation is optional.

$(DOC_IMPLEMENTED_METHODS(:intercept)).

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

$(DOC_IMPLEMENTED_METHODS(:tree)).

"""
function tree end

"""
    LearnAPI.trees(model)

For some ensemble model, return a vector of trees. See [`LearnAPI.tree`](@ref) for the
form of such trees.

See also [`LearnAPI.tree`](@ref).

# New implementations

Implementation is optional.

$(DOC_IMPLEMENTED_METHODS(:trees)).

"""
function trees end

"""
    LearnAPI.training_losses(model)

Return the training losses obtained when running `model = fit(algorithm, ...)` for some
`algorithm`.

See also [`fit`](@ref).

# New implementations

Implement for iterative algorithms that compute and record training losses as part of
training (e.g. neural networks).

$(DOC_IMPLEMENTED_METHODS(:training_losses)).

"""
function training_losses end

"""
    LearnAPI.training_scores(model)

Return the training scores obtained when running `model = fit(algorithm, ...)` for some
`algorithm`.

See also [`fit`](@ref).

# New implementations

Implement for algorithms, such as outlier detection algorithms, which associate a score
with each observation during training, where these scores are of interest in later
processes (e.g, in defining normalized scores for new data).

$(DOC_IMPLEMENTED_METHODS(:training_scores)).

"""
function training_scores end

"""
    LearnAPI.components(model)

For a composite `model`, return the component models (`fit` outputs). These will be in the
form of a vector of named pairs, `property_name::Symbol => component_model`. Here
`property_name` is the name of some algorithm-valued property (hyper-parameter) of
`algorithm = LearnAPI.algorithm(model)`.

A composite model is one for which the corresponding `algorithm` includes one or more
algorithm-valued properties, and for which `LearnAPI.is_composite(algorithm)` is `true`.

See also [`is_composite`](@ref).

# New implementations

Implementent if and only if `model` is a composite model. 

$(DOC_IMPLEMENTED_METHODS(:components)).

"""
function components end

"""
    LearnAPI.training_labels(model)

Return the training labels obtained when running `model = fit(algorithm, ...)` for some
`algorithm`.

See also [`fit`](@ref).

# New implementations

$(DOC_IMPLEMENTED_METHODS(:training_labels)).

"""
function training_labels end


# :extras intentionally excluded:
const ACCESSOR_FUNCTIONS_WITHOUT_EXTRAS = (
    algorithm,
    coefficients,
    intercept,
    tree,
    trees,
    feature_importances,
    training_labels,
    training_losses,
    training_scores,
    components,
)

const ACCESSOR_FUNCTIONS_WITHOUT_EXTRAS_LIST = join(
    map(ACCESSOR_FUNCTIONS_WITHOUT_EXTRAS) do f
    "[`LearnAPI.$f`](@ref)"
    end,
    ", ",
    " and ",
)

 """
    LearnAPI.extras(model)

Return miscellaneous byproducts of an algorithm's computation, from the object `model`
returned by a call of the form `fit(algorithm, data)`.

$DOC_STATIC

See also [`fit`](@ref).

# New implementations

Implementation is discouraged for byproducts already covered by other LearnAPI.jl accessor
functions: $ACCESSOR_FUNCTIONS_WITHOUT_EXTRAS_LIST.

$(DOC_IMPLEMENTED_METHODS(:training_labels)).

"""
function extras end

const ACCESSOR_FUNCTIONS = (extras, ACCESSOR_FUNCTIONS_WITHOUT_EXTRAS...)

const ACCESSOR_FUNCTIONS_LIST = join(
    map(ACCESSOR_FUNCTIONS) do f
    "[`LearnAPI.$f`](@ref)"
    end,
    ", ",
    " and ",
)

