# Common Implementation Patterns

!!! warning

	This section is only an implementation guide. The definitive specification of the
	Learn API is given in [Reference](@ref reference).

This guide is intended to be consulted after reading [Anatomy of an Implementation](@ref),
which introduces the main interface objects and terminology.

Although an implementation is defined purely by the methods and traits it implements, most
implementations fall into one (or more) of the following informally understood patterns or
"tasks":

- [Classifiers](@ref): Supervised learners for categorical targets

- [Regressors](@ref): Supervised learners for continuous targets

- [Iterative Algorithms](@ref)

- [Incremental Algorithms](@ref)

- [Static Transformers](@ref): Transformations that do not learn but which have
  hyperparameters and/or deliver ancillary information about the transformation

- [Dimension Reduction](@ref): Transformers that learn to reduce feature space dimension

- [Missing Value Imputation](@ref): Transformers that replace missing values.

- [Clusterering](@ref): Algorithms that group data into clusters for classification and
  possibly dimension reduction. May be true learners (generalize to new data) or static.

- [Outlier Detection](@ref): Supervised, unsupervised, or semi-supervised learners for
  anomaly detection.

- [Learning a Probability Distribution](@ref): Algorithms that fit a distribution or
  distribution-like object to data

- [Time Series Forecasting](@ref)

- [Time Series Classification](@ref)

- [Supervised Bayesian Algorithms](@ref)

- [Survival Analysis](@ref)

