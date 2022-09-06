```@raw html
<script async defer src="https://buttons.github.io/buttons.js"></script>
<span style="color: #9558B2;font-size:4.5em;">
LearnAPI.jl</span>
<br>
<span style="color: #9558B2;font-size:1.6em;font-style:italic;">
A basic Julia interface for training and applying machine learning models </span>
<br><br>
```

**Quick tour for developers of ML software.** For a rapid overview, by way of a sample
implementation, see [Anatomy of an Implementation](@ref).

**Quick tour for users of models implementing LearnAPI.jl.** Although primarily intended
as a basement-level machine learning interface for developers, users can interact directly
with LearnAPI.jl models, as illustrated [here](@ref workflow). 

## Approach

Machine learning algorithms, also called *models*, have a complicated taxonomy. Grouping
models, or modelling tasks, into a relatively small number of types, such as "classifier"
and "clusterer", and attempting to impose uniform behaviour within each group, is
challenging. In our experience developing the [MLJ
ecosystem](https://github.com/alan-turing-institute/MLJ.jl), this either leads to
limitations on the models that can be included in a general interface, or additional
complexity needed to cope with exceptional cases. Even if a complete user interface for
machine learning might benefit from such groupings, a basement-level API for ML should, in
our view, avoid them.

## Summary

LearnAPI.jl is a base interface for machine learning algorithms in which behaviour is
articulated using traits. It has no abstract model types, apart from an optional supertype
`Model`. It provides the following methods, dispatched on model type:

- `fit` for regular training, overloaded if the model generalizes to new data, as in
  classical supervised learning

- `update!` for adding model iterations, or responding efficiently to other
  post-`fit`changes in hyperparameters

- `ingest!` for incremental learning

- **operations**, such as `predict`, `transform` and `inverse_transform` for applying the
  model to data not used for training

- common **access functions**, such as `feature_importances` and `training_losses`, for
  extracting, from training outcomes information, common to particular classes of models

- **model traits**, such as `target_proxy_kind(model)`, for promising specific behaviour

There is flexibility about how much of the interface is implemented by a given model
object `model`. A special trait `implemented_methods(model)` declares what has been
explicitly implemented or overloaded to work with `model`.

Since this is a functional-style interface, `fit` returns model `state`, in addition to
learned parameters, for passing to the optional `update!` and `ingest!` methods. These
training methods also return a `report` component, for exposing byproducts of training
different from learned parameters. Similarly, all operations also return a `report`
component (important for models that do not generalize to new data).

Models can be supervised or not supervised, can generalize to new data observations, or
not generalize. To ensure proper handling by client packages of probabilistic and other
non-literal forms of target predictions (pdfs, confidence intervals, survival functions,
etc) the kind of prediction can be flagged appropriately; see more at "target" below.


## [Scope and undefined notions](@id scope)

The Learn API provides methods for training, applying, and saving machine learning models,
and that is all. *It does not specify an interface for data access or data
resampling*. However, LearnAPI.jl is predicated on a few basic undefined notions (in
**boldface**) which some higher-level interface might decide to formalize:

- An object which generates ordered sequences of individual **observations** is
  called **data**.

- Each machine learning model's behaviour is governed by a number of user-specified
  **hyperparameters**.

- Information needed for training that is not a model hyperparameter and not data is called
  **metadata** (e.g., target class weights and group lasso feature groupings).

- Some models involve the notion of a **target** variable and generate output with the
  same form as the target, or, more generally, some kind of target proxy, such as
  probability distributions. A *target proxy* is something that can be *paired* with target
  data to obtain useful information about the model and the data that has been presented
  to it, typically a measure of the model's expected performance on unseen data. A target
  variable is not necessarily encountered during training, i.e., target variables can make
  sense for unsupervised models, and also for models that do not generalize to new
  observations.  For examples, and an informal classification of target proxy types, refer
  to [Target proxies](@ref).
  

## Contents

Our opening observations notwithstanding, it is useful to have a guide to the interface,
linked below, organized around common *informally defined* patterns or "tasks". However,
the definitive specification of the interface is the [Reference](@ref) section.

- [Anatomy of an Implementation](@ref) (Overview)

- [Common Implementation Patterns](@ref) (User Guide)

- [Reference](@ref)

- [Testing an Implementation](@ref)

!!! info

	It is strongly recommended users read  [Anatomy of an Implementation](@ref) before
	consulting the guide or reference sections.


**Note.** In the future, LearnAPI.jl will become the new foundation for the
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) toolbox created by the same
developers. However, LearnAPI.jl is meant as a general purpose, standalone, lightweight
API for machine learning algorithms (and has no reference to the "machines" used there).
