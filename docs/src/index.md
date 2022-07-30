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
attempting to impose uniform behaviour within each group, is a problematic approach. It
either leads to limitations on the models that can be included in a general interface, or to
undesirable complexity needed to cope with exceptional cases.

For these and other reasons, the behaviour of a model implementing the **ML Model
Interface** documented here is articulated using traits - methods dispatched on
the model type, such as `is_supervised(model::SomeModel) = true`. There are a small number
of compulsory traits and a larger number of optional ones.  There is a single abstract model
type `Model`, but model types can implement the interface without subtyping this. There is
no abstract model type hierarchy.

The preceding observations notwithstanding, it is useful to have a guide to the interface
organized around common informally defined patterns; the definitive specification of the
interface is provided in the [Reference](@ref) section:

- Overview: [Anatomy of an Implementation](@ref)

- User Guide: [Common Implementation Patterns](@ref)

- [Reference](@ref)

- [Testing an implementation](@ref)

!!! info

	It is strongly recommended users read  [Anatomy of an Implementation](@ref) before
	consulting the guide or reference sections.


**Note.** The ML Model Interface provides a foundation for the
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/)'s "machine" interface for user
interaction. However it is a general purpose, standalone, lightweight API for machine
learning algorithms (and has no reference to machines).


