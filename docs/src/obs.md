# [`obs`](@id data_interface)

The [MLUtils.jl](https://github.com/JuliaML/MLUtils.jl) package provides two methods
`getobs` and `numobs` for resampling data divided into multiple observations, including
arrays and tables. The data objects returned below are guaranteed to implement this
interface and can be passed to the relevant method (`obsfit`, `obspredict` or
`obstransform`) possibly after resampling using `MLUtils.getobs`. This may provide
performance advantages over naive workflows.

```julia
obs(fit, algorithm, data...) -> <combined data object for fit>
obs(predict, algorithm, data...) -> <combined data object for predict>
obs(transform, algorithm, data...) -> <combined data object for transform>
```

## Typical workflows

LearnAPI.jl makes no assumptions about the form of data `X` and `y` in a call like
`fit(algorithm, X, y)`. The particular `algorithm` is free to articulate it's own
requirements.  However, in this example, the definition

```julia
obsdata = obs(fit, algorithm, X, y)
```

combines `X` and `y` in a single object guaranteed to implement the MLUtils.jl
`getobs`/`numobs` interface, which can be passed to `obsfit` instead of `fit`, as is, or
after resampling using `MLUtils.getobs`:

```julia
# equivalent to `fit(algorithm, X, y)`:
model = obsfit(algorithm, obsdata)

# with resampling:
resampled_obsdata = MLUtils.getobs(obsdata, 1:100)
model = obsfit(algorithm, resampled_obsdata)
```

In some implementations, the alternative pattern above can be used to avoid repeating
unnecessary internal data preprocessing, or inefficient resampling.  For example, here's
how a user might call `obs` and `MLUtils.getobs` to perform efficient
cross-validation:

```julia
using LearnAPI
import MLUtils

X = <some data frame with 30 rows>
y = <some categorical vector with 30 rows>
algorithm = <some LearnAPI-compliant algorithm>

test_train_folds = map([1:10, 11:20, 21:30]) do test
    (test, setdiff(1:30, test))
end 

# create fixed model-specific representations of the whole data set:
fit_data = obs(fit, algorithm, X, y)
predict_data = obs(predict, algorithm, predict, X)

scores = map(train_test_folds) do (train_indices, test_indices)
    
	# train using model-specific representation of data:
	train_data = MLUtils.getobs(fit_data, train_indices)
	model = obsfit(algorithm, train_data)
	
	# predict on the fold complement:
	test_data = MLUtils.getobs(predict_data, test_indices)
	ŷ = obspredict(model, LiteralTarget(), test_data)

    return <score comparing ŷ with y[test]>
	
end 
```

Note here that the output of `obspredict` will match the representation of `y` , i.e.,
there is no concept of an algorithm-specific representation of *outputs*, only inputs.


## Implementation guide

| method        | compulsory? | fallback               |
|:--------------|:-----------:|:----------------------:|
| [`obs`](@ref) | depends     | slurps `data` argument |
|               |             |                        |

If the `data` consumed by `fit`, `predict` or `transform` consists only of tables and
arrays (with last dimension the observation dimension) then overloading `obs` is
optional. However, if an implementation overloads `obs` to return a (thinly wrapped)
representation of user data that is closer to what the core algorithm actually uses, and
overloads `MLUtils.getobs` (or, more typically `Base.getindex`) to make resampling of that
representation efficient, then those optimizations become available to the user, without
the user concerning herself with the details of the representation.

A sample implementation is given in the [`obs`](@ref) document-string below.

```@docs
obs
```

