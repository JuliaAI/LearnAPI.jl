```@raw html
<script async defer src="https://buttons.github.io/buttons.js"></script>
<span style="color: #9558B2;font-size:4.5em;">
LearnAPI.jl</span>
<br>
<span style="color: #9558B2;font-size:1.6em;font-style:italic;">
A base Julia interface for machine learning and statistics </span>
<br><br>
```

## Accelerated overview

LearnAPI.jl provides a collection methods stubs, such as `fit` and `predict`, to be
implemented by algorithms from machine learning and statistics. Through such
implementations, such algorithms buy into algorithm-generic functionality, such as
hyperparameter optimization, as provided by ML/statistics toolboxes and other
packages. LearnAPI.jl also provides a number of Julia traits for making specific promises
of behavior.

It is designed to be powerful, from the point of view of adding algorithm-generic
functionality, while minimizing the burden on developers implementing the API for a
specific algorithm.

- To see how to **DEVELOPERS INTERACT** with algorithms implementing LearnAPI, see [Basic fit/predict
  workflow](@ref workflow).

- To see how **USERS INTERACT** with LearnAPI algorithms, see [User
  Interface](@ref).[under construction]

- For developers wanting to **IMPLEMENT** LearnAPI, see [Anatomy of
  an Implementation](@ref).

For more on package goals and philosophy, see [Goals and Approach](@ref).


## Methods

In LearnAPI an *algorithm* is a Julia object storing the hyperparameters of some
ML/statistics algorithm.

The following methods, dispatched on algorithm type, are provided:

- `fit`, overloaded if an algorithm involves a learning step, as in classical supervised
  learning; the principal output of `fit` is learned parameters

- `update!`, for adding iterations to an algorithm, or responding efficiently to other
  post-`fit`changes in hyperparameters

- `ingest!`, for incremental learning (training further using *new* data, without
  re-initializing learned parameters)

- *operations*, which apply the algorithm to data, typically not seen in
  training, if there is any:

  - `predict`, for predicting values of a target variable or a proxy for the target, such	as probability distributions; see below

  - `transform`, for other kinds transformations

  - `inverse_transform`, for reconstructing data from a transformed representation

- common *accessor functions*, such as `feature_importances` and `training_losses`, for
  extracting, from training outcomes, information common to a number of different
  algorithms

- *algorithm traits*, such as `predict_output_type(algorithm)`, for promising specific behavior

Since this is a functional-style interface, `fit` returns algorithm `state`, in addition to
learned parameters, for passing to the optional `update!` and `ingest!` methods. These
training methods also return a `report` component, for exposing byproducts of training
different from learned parameters. Similarly, all operations also return a `report`
component (important for algorithms that do not generalize to new data).


## [Informal concepts](@id scope)

LearnAPI.jl is predicated on a few basic, informally defined notions, in *italics*
below, which some higher-level interface might decide to formalize.

- An object which generates ordered sequences of individual *observations* is called
  *data*. For example a `DataFrame` instance, from
  [DataFrames.jl](https://dataframes.juliadata.org/stable/), is considered data, the
  observations being the rows. A matrix can be considered data, but whether the
  observations are rows or columns is ambiguous and not fixed by LearnAPI.

- Each machine learning algorithm's behavior is governed by a number of user-specified
  *hyperparameters*. The regularization parameter in ridge regression is an
  example. Hyperparameters are data-independent. For example, the number of target classes
  is not a hyperparameter.

- Information needed for training that is not a hyperparameter and not data is called
  *metadata*. Examples, include target *class* weights and group lasso feature
  groupings. Further examples include feature names, and the pool of target classes, when
  these are not embedded in the data representation.


### [Targets and target proxies](@id proxy)

After training, a supervised classifier predicts labels on some input which are then
compared with ground truth labels using some accuracy measure, to assesses the performance
of the classifier. Alternatively, the classifier predicts class probabilities, which are
instead paired with ground truth labels using a proper scoring rule, say. In outlier
detection, "outlier"/"inlier" predictions, or probability-like scores, are similarly
compared with ground truth labels. In clustering, integer labels assigned to observations
by the clustering algorithm can can be paired with human labels using, say, the Rand
index. In survival analysis, predicted survival functions or probability distributions are
compared with censored ground truth survival times.

More generally, whenever we have a predicted variable (e.g., a class label) paired with
itself or some proxy (such as a class probability) we call the variable a *target*
variable, and the predicted output a *target proxy*. It is immaterial whether or not the
target appears in training (is supervised) or whether the model generalizes to new
observations (learns) or not. 

The target and the kind of predicted proxy are crucial features of ML/statistics
performance measures and LearnAPI.jl provides a detailed list of proxy dispatch types (see
[Target proxies](@ref)), as well as algorithm traits to articulate target type /scitype.


## Optional data interface

It can be useful to distinguish between data that exists at some high level, convenient
for the general user - such as a table (dataframe) or the path to a directory containing
image files - and a performant, algorithm-specific representation of that data, such as a
matrix or image "data loader". When retraining using the same data with new
hyperparameters, one wants to avoid recreating the algorithm-specific representation, and,
accordingly, a higher level interface may want to cache such representations. Furthermore,
in resampling (e.g., cross-validation), a higher level interface wants to directly
resample the algorithm-specific representation, so it needs to know how to do that. To
meet these two ends, LearnAPI provides two additional *data methods* dispatched on
algorithm type:

- `reformat(algorithm, ...)`, for converting from a user data representation to a
  performant algorithm-specific representation, whose output is for use in `fit`,
  `predict`, etc. above

- `getobs(algorithm, ...)`, for extracting a subsample of observations of the
  algorithm-specific representation

It should be emphasized that LearnAPI is itself agnostic to particular representations of
data or the particular methods of accessing observations within them. By overloading these
methods, each `algorithm` is free to choose its own data interface.

See [Optional data Interface](@ref data_interface) for more details.

## Contents

It is useful to have a guide to the interface, linked below, organized around common
*informally defined* patterns or "tasks". However, the definitive specification of the
interface is the [Reference](@ref reference) section.

- Overview: [Anatomy of an Implementation](@ref)

- Official Specification: [Reference](@ref reference)

- User guide: [Common Implementation Patterns](@ref) [under construction]

- [Testing an Implementation](@ref) [under construction]

!!! info

	It is recommended developers read [Anatomy of an Implementation](@ref) before
	consulting the guide or reference sections.

*Note.* In the future, LearnAPI.jl may become the new foundation for the
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) toolbox. However, LearnAPI.jl
is meant as a general purpose, stand-alone, lightweight, low level API (and has no
reference to the "machines" used in MLJ).
