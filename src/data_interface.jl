"""
    LearnAPI.getobs(model, LearnAPI.fit, I, data...)

Return a subsample of `data` consisting of all observations with indices in `I`. Here
`data` is data of the form expected in a call like `LearnAPI.fit(model, verbosity,
data...; metadata...)`.

Always returns a tuple of the same length as `data`.

    LearnAPI.getobs(model, operation, I, data...)

Return a subsample of `data` consisting of all observations with indices in `I`. Here
`data` is data of the form expected in a call of the specified `operation`, e.g., in a
call like `LearnAPI.predict(model, data...)`, if `operation = LearnAPI.predict`. Possible
values for `operation` are: $DOC_OPERATIONS_LIST.

Always returns a tuple of the same length as `data`.

# New model implementations

Implementation is optional. If implemented, then ordinarily implemented for each signature
of `fit` and operation implemented for `model`.

$(DOC_IMPLEMENTED_METHODS(:reformat))

The subsample returned must be acceptable in place of `data` in the call function named in
the second argument.

## Example implementation

Suppose that `MyClassifier` is a model type for simple supervised classification, with
`LearnAPI.fit(model::MyClassifier, verbosity, A, y)` and `predict(model::MyClassifier,
fitted_params, A)` implemented assuming the target `y` is an ordinary abstract vector and
the features `A` is an abstract matrix with columns as observations. Then the following is
a valid implementation of `getobs`:

```julia
LearnAPI.getobs(::MyClassifier, ::typeof(LearnAPI.fit), I, A, y) =
    (view(A, :, I), view(y, I))
LearnAPI.getobs(::MyClassifier, ::typeof(LearnAPI.predict), I, A) = (view(A, :, I),)
```

"""
function getobs end

"""
    LearnAPI.reformat(model, LearnAPI.fit, user_data...; metadata...)

Return the model-specific representations `(data, metadata)` of user-supplied `(user_data,
user_metadata)`, for consumption, after splatting, by `LearnAPI.fit`, `LearnAPI.update!`
or `LearnAPI.ingest!`.

    LearnAPI.reformat(model, operation, user_data...)

Return the model-specific representation `data` of user-supplied `user_data`, for
consumption, after splatting, by the specified `operation`, dispatched on `model`. Here
`operation` is one of: $DOC_OPERATIONS_LIST.

The following sample workflow illustrates the use of both versions of `reformat`above:

```julia
data, metadata = LearnAPI.reformat(model, LearnAPI.fit, X, y; class_weights=dic)
fitted_params, state, fit_report = LearnAPI.fit(model, 0, data...; metadata...)

test_data = LearnAPI.reformat(model, LearnAPI.predict, Xtest)
ŷ, predict_report = LearnAPI.predict(model, fitted_params, test_data...)
```

# New model implementations

Implementation of `reformat` is optional. The fallback simply slurps the supplied
data/metadata. You will want to implement for each `fit` or operation signature
implemented for `model`.

$(DOC_IMPLEMENTED_METHODS(:reformat, overloaded=true))

Ideally, any potentially expensive transformation of user-supplied data that is carried
out during training only once, at the beginning, should occur in `reformat` instead of
`fit`/`update!`/`ingest!`.

Note that the first form of `reformat`, for operations, should always return a tuple,
because the output is splat in calls to the operation (see the sample workflow
above). Similarly, in the return value `(data, metadata)` for the `fit` variant, `data` is
always a tuple and `metadata` always a named tuple (or `Base.Pairs` object). If there is
no metadata, a `NamedTuple()` can be returned in its place.

## Example implementation

Suppose that `MyClassifier` is a model type for simple supervised classification, with
`LearnAPI.fit(model::MyClassifier, verbosity, A, y; names=...)` and
`predict(model::MyClassifier, fitted_params, A)` implemented assuming that the target `y`
is an ordinary vector, the features `A`is a matrix with columns as observations, and
`names` are the names of the features. Then, supposing users supply features in tabular
form, but target as expected, then we provide the following implementation of `reformat`:

```julia
using Tables
function LearnAPI.reformat(::MyClassifier, ::typeof(LearnAPI.fit), X, y)
    names = Tables.schema(Tables.rows(X)).names
    return ((Tables.matrix(X)', y), (; names))
end
LearnAPI.reformat(::MyClassifier, ::typeof(LearnAPI.predict), X) = (Tables.matrix(X)',)
```
"""
reformat(::Any, ::Any, data...; model_data...) = (data, model_data)
