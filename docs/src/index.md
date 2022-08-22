```@raw html
<script async defer src="https://buttons.github.io/buttons.js"></script>
<span style="color: #9558B2;font-size:4.5em;">
LearnAPI.jl</span>
<br>
<span style="color: #9558B2;font-size:1.6em;font-style:italic;">
A Julia interface for training and applying models in machine learning and statistics</span>
```

Machine learning algorithms, also called *models*, have a complicated taxonomy. Grouping
models into a relatively small number of types, such as "classifier" and "clusterer", and
attempting to impose uniform behaviour within each group, is challenging. In our
experience, it either leads to limitations on the models that can be included in a general
interface, or additional complexity needed to cope with exceptional cases. Even if a
complete user interface for machine learning might benefit from such groupings, a
basement-level API for ML should, in our view, avoid them.

The **Learn API** documented here is base API for machine learning that is purely
functional with no abstract model types (apart an optional supertype `Model`). It provides
the following methods, dispatched on model type:

- `fit` for regular training

- `update!` for adding model iterations, or responding efficiently to other
  post-`fit`changes in hyperparameters

- `ingest!` for incremental learning

- **operations**, such as `predict`, `transform` and `inverse_transform` for applying the model
  to data

- common **access functions**, such as `feature_importances` and `training_losses`, for
  extracting from training outcomes information common to particular classes of models.

- **model traits**, such as `is_supervised(model)`, for promising specific behaviour.

Since this is a functional interface, `fit` returns model "state", in addition to learned
parameters, for passing to the optional `update!` and `ingest!` methods. These three
methods all return a `report` component, for exposing byproducts of training different
from learned parameters. Similarly, all operations also return a `report` component,
although this would typically be `nothing`, unless the model does not implement `fit`
(does not generalize to new data).


## Scope and undefined notions

The Learn API provides methods for training, applying, and saving machine learning models,
and that is all. To keep it *It does not specify an interface for data access or data
resampling*. That said, the interface references a few basic undefined notions, which some
higher-level interface might decide to formalize:

- Each machine learning model's behaviour is governed by a number of user-specified
  **hyper-parameters**.

- An object which generates ordered sequences of individual **observations** is called
  **data**.

- Information needed for training that is not a model hyper-parameter and not data is called
  **metadata** (e.g., target class weights and group lasso feature groupings).

- Some models, including but not limited to supervised models, involve **target** data, in
  training or otherwise, and implement an operation, typically `predict`, that outputs
  data that is target-like. To say that data is **target-like** is to say that it can be
  paired with target data having the same number of observations to obtain useful
  information about the model and the data that has been presented to it, typically a
  measure of the model's expected performance on unseen data. Target-like data can take
  various informally defined forms, such as `Deterministic`, `Distribution`, `Sampleable`,
  `SurvivalFunction` and `Interval` detailed further under [Operations](@ref operations).

Regarding the last point, consider outlier detection, where target observations are either
"outlier" or "inlier". If the detector predicts probabilities for outlierness (the
target-like data) these can be paired with "outlier"/"inlier" labels assigned by humans,
using, say, area under the ROC curve, to measure performance. Many such detectors are
trainined without supervision.


## Contents

Our opening observations notwithstanding, it is useful to have a guide to the interface,
linked below, organized around common *informally defined* patterns or "tasks". However,
the definitive specification of the interface is the [Reference](@ref) section.

- [Anatomy of an Implementation](@ref) (Overview)

- [Common Implementation Patterns](@ref) (User Guide)

- [Reference](@ref)

- [Testing an implementation](@ref)

!!! info

	It is strongly recommended users read  [Anatomy of an Implementation](@ref) before
	consulting the guide or reference sections.


**Note.** The Learn API provides a foundation for the higher level "machine"
interface for user interaction in the toolbox
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) created by the same
developers. However, the Learn API provided here is meant as a general purpose,
standalone, lightweight API for machine learning algorithms (and has no reference to
machines).
