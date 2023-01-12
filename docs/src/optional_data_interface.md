# [Optional Data Interface](@id data_interface)

## Resampling

To aid in programmatic resampling, such as cross-validation, it is helpful if each machine
learning model articulates how the data it consumes can be subsampled - that is, how a
subset of observations can be extracted from that data. Another advantage of doing so is to
mitigate some of the ambiguities around structuring observations within the container (are
the observations in a matrix the rows or the columns?).

In LearnAPI, an implementation can articulate a subsampling method by implementing
`LearnAPI.getobs(model, func, I, data...)` for each function `func` consuming data, such
as `fit` and `predict`. Examples are given below.

```@docs
LearnAPI.getobs
```
## Preprocessing

So that a higher level interface can avoid unnecessarily repeating calls to convert
user-supplied data (e.g., a dataframe) into some performant, model-specific
representation, a model can move such data conversions out of `fit`, `predict`, etc., and
into an implementation of `LearnAPI.reformat` created for each signature of such methods
that are implemented. Examples are given below.

```@docs
LearnAPI.reformat
```

