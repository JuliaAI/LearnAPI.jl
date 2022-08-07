```@raw html
<script async defer src="https://buttons.github.io/buttons.js"></script>
<span style="color: #9558B2;font-size:4.5em;">
MLInterface.jl</span>
<br>
<span style="color: #9558B2;font-size:1.6em;font-style:italic;">
A Julia interface for training and applying models in machine learning and statistics</span>
```

# Introduction

Machine learning algorithms, also called *models*, have a complicated taxonomy. Grouping
models into a relatively small number of types, such as "classifier" and "clusterer", and
attempting to impose uniform behaviour within each group, is problematic. It either leads to
limitations on the models that can be included in a general interface, or to undesirable
complexity needed to cope with exceptional cases.

For these and other reasons, the **ML Model Interface** documented here is purely functional
with no abstract model types (apart an optional supertype `Model`). In addition to `fit`,
`update!` and `ingest!` methods (all optional), one implements one or more operations, such
as `predict`, `transform` and `inverse_transform`. Method stubs for access functions, such
as `feature_importances`, are also provided. Finally, a number of optional trait
declarations, such as `is_supervised(model::SomeModel) = true`, make promises of specific
behaviour.

The ML Model Interface provides methods for training, applying, and saving machine learning
models, and that is all. It does not provide an interface for data resampling, although it
informally distinguishes between training data consisting of "observations", and other
"metadata", such as target class weights or group lasso feature gropings. At present the
only restriction on data containers concerns the target predictions of supervised models
(whether deterministic, probabilistic or otherwise): These must be abstract arrays or tables
compatible with [Tables.jl](https://github.com/JuliaData/Tables.jl).

Our opening observations notwithstanding, it is useful to have a guide to the interface,
linked below, organized around common *informally defined* patterns. However, the definitive
specification of the interface is the [Reference](@ref) section.

- [Anatomy of an Implementation](@ref) (Overview)

- [Common Implementation Patterns](@ref) (User Guide)

- [Reference](@ref)

- [Testing an implementation](@ref)

!!! info

	It is strongly recommended users read  [Anatomy of an Implementation](@ref) before
	consulting the guide or reference sections.


**Note.** The ML Model Interface provides a foundation for the higher level "machine"
interface for user interaction in the toolbox
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) created by the same
developers. However, the ML Model Interface provided here is meant as a general purpose,
standalone, lightweight API for machine learning algorithms (and has no reference to
machines).
