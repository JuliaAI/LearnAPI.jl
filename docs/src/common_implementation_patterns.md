# Common Implementation Patterns

!!! warning

	This section is only an implementation guide. The definitive specification of the
	Learn API is given in [Reference](@ref).

This guide is intended to be consulted after reading [Anatomy of a Model
Implementation](@ref), which introduces the main interface objects and terminology.

Although an implementation is defined purely by the methods and traits it implements, most
implementations fall into one (or more) of the following informally understood patterns or
"tasks":

- [Classifiers](@ref): Supervised learners for categorical targets

- [Regressors](@ref): Supervised learners for continuous targets

- [Iterative Models](@ref)

- [Incremental Models](@ref)

- [Static Transformers](@ref): Transformations that do not learn but which have
  hyper-parameters and/or deliver ancilliary information about the transformation

- [Dimension Reduction](@ref): Transformers that learn to reduce feature space dimension

- [Missing Value Imputation](@ref): Transformers that replace missing values.

- [Clusterering](@ref): Algorithms that group data into clusters for classification and
  possibly dimension reduction. May be true learners (generalize to new data) or static.

- [Outlier Detection](@ref): Supervised, unsupervised, or semi-supervised learners for
  anomaly detection.

- [Learning a Probability Distribution](@ref): Models that fit a distribution or
  distribution-like object to data

- [Time Series Forecasting](@ref)

- [Time Series Classification](@ref)

- [Supervised Bayesian Models](@ref)

- [Survival Analysis](@ref)

