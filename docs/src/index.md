```@raw html
<script async defer src="https://buttons.github.io/buttons.js"></script>
<span style="color: #9558B2;font-size:4.5em;">
LearnAPI.jl</span>
<br>
<span style="color: #9558B2;font-size:1.6em;font-style:italic;">
A basement-level Julia interface for training and applying machine learning models </span>
<br><br>
```

## Accelerated overview

LearnAPI provides a collection methods stubs, such as `fit` and `predict`, to be
implemented by machine learning (ML) models wishing to buy into model-generic
functionality, such as hyperparameter optimization, provided by ML toolboxes and other
packages. It also provides a number of Julia traits for making specific promises of
behaviour. 

It is designed to be powerful, from the point of view of adding model-generic
functionality, while minimising the burden on devlopers implementing the API for a
specific model.

- To see how to **INTERACT WITH** models implementing LearnAPI: [Basic fit/predict
  workflow](@ref workflow).

- For developers wanting to **IMPLEMENT** LearnAPI: [Anatomy of
  an Implementation](@ref).

For more on package goals and philosophy, see [Goals and Approach](@ref).


## Methods

In LearnAPI.jl a *model* is just a container for the hyper-parameters of some machine
learning algorithm, and does not typically include learned parameters.

The following methods, dispatched on model type, are provided:

- `fit`, for regular training, overloaded if the model generalizes to new data, as in
  classical supervised learning; the principal output of `fit` is the learned parameters

- `update!`, for adding model iterations, or responding efficiently to other
  post-`fit`changes in hyperparameters

- `ingest!`, for incremental learning (training further using *new* data, without
  re-initializing learned parameters)

- **operations**, `predict`, `predict_joint`, `transform` and `inverse_transform` for
  applying the model to data not used for training

- common **accessor functions**, such as `feature_importances` and `training_losses`, for
  extracting, from training outcomes, information common to some models

- **model traits**, such as `predict_output_type(model)`, for promising specific behavior

There is flexibility about how much of the interface is implemented by a given model type.
A special trait `functions(model)` declares what has been explicitly implemented to work
with `model`, excluding traits.

Since this is a functional-style interface, `fit` returns model `state`, in addition to
learned parameters, for passing to the optional `update!` and `ingest!` methods. These
training methods also return a `report` component, for exposing byproducts of training
different from learned parameters. Similarly, all operations also return a `report`
component (important for models that do not generalize to new data).


## [Scope, undefined notions](@id scope)

The basic LearnAPI.jl interface provides methods for training and applying machine
learning models, and that is all. The interface specification is predicated on a few basic
undefined notions (in **boldface**) which some higher-level interface might decide to
formalize. 

- An object which generates ordered sequences of individual **observations** is called
  **data**. For example a `DataFrame` instance, from
  [DataFrames.jl](https://dataframes.juliadata.org/stable/), is considered data, the
  observations being the rows. A matrix can be considered data, but whether the
  observations are rows or columns is ambiguous and not fixed by LearnAPI.

- Each machine learning model's behavior is governed by a number of user-specified
  **hyperparameters**. The regularization parameter in ridge regression is an
  example. Hyperparameters are data-independent. For example, the number of target classes
  is not a hyperparameter.

- Information needed for training that is not a model hyperparameter and not data is
  called **metadata**. Examples, include target *class* weights, group lasso feature
  groupings. Other examples are feature names and the pool of target classes, when these
  are not embedded in the data representation.


### Targets and target proxies

Some models involve the notion of a **target** variable, and generate output with the same
form as the target, or, more generally, some kind of target proxy, such as probability
distributions, or survival functions. A *target proxy* is something that can be *paired*
with target data to obtain useful information about the model and the data that has been
presented to it, typically a measure of the model's expected performance on unseen
data. A target variable is not necessarily encountered during training, i.e., target
variables can make sense for unsupervised models, and also for models that do not
generalize to new observations.  For examples, and an informal classification of target
proxy types, refer to [Target proxies](@ref).


## Optional data interface

It can be useful to distinguish between data that exists at some high level, convenient
for the general user - such as a table (dataframe) or the path to a directory containing
image files - and a performant, model-specific representation of that data, such as a
matrix or image "data loader". When retraining using the same data with new
hyper-parameters, one wants to avoid recreating the model-specific representation, and,
accordingly, a higher level ML interface may want to cache such
representations. Furthermore, in resampling (e.g., performing cross-validation), a higher
level interface wants only to resample the model-specific representation, so it needs to
know how to do that. To meet these two ends, LearnAPI provides two additional **data
methods** dispatched on model type:

- `reformat(model, ...)`, for converting from a user data representation to a performant model-specific representation, whose output is for use in `fit`, `predict`, etc. above

- `getobs(model, ...)`, for extracting a subsample of observations of the model-specific
  representation

It should be emphasized that LearnAPI is itself agnostic to particular representations of
data or the particular methods of accessing observations within them. By overloading these
methods, each `model` is free to choose its own data interface.

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

	It is recommended users read  [Anatomy of an Implementation](@ref) before
	consulting the guide or reference sections.

**Note.** In the future, LearnAPI.jl may become the new foundation for the
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) toolbox created by the same
developers. However, LearnAPI.jl is meant as a general purpose, stand-alone, lightweight,
low level API for machine learning algorithms (and has no reference to the "machines" used
there).
