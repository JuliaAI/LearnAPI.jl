# [Optional Data Interface](@id data_interface)

> **Summary.** Implement `getobs` to articulate how to generate individual observations
> from data consumed by a LearnAPI algorithm. Implement `reformat` to provide a higher level
> interface the means to avoid repeating transformations from user representations of data
> (such as a dataframe) and algorithm-specific representations (such as a matrix).

## Resampling

To aid in programmatic resampling, such as cross-validation, it is helpful if each machine
learning algorithm articulates how the data it consumes can be subsampled - that is, how a
subset of observations can be extracted from that data. Another advantage of doing so is
to mitigate some of the ambiguities around structuring observations within the container:
Are the observations in a matrix the rows or the columns?

In LearnAPI, an implementation can articulate a subsampling method by implementing
`LearnAPI.getobs(algorithm, func, I, data...)` for each function `func` consuming data, such
as `fit` and `predict`. Examples are given below.

```@docs
LearnAPI.getobs
```
## Preprocessing

So that a higher level interface can avoid unnecessarily repeating calls to convert
user-supplied data (e.g., a dataframe) into some performant, algorithm-specific
representation, an algorithm can move such data conversions out of `fit`, `predict`, etc., and
into an implementation of `LearnAPI.reformat` created for each signature of such methods
that are implemented. Examples are given below.

```@docs
LearnAPI.reformat
```

