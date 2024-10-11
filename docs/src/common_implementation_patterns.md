# Common Implementation Patterns

```@raw html
&#128679;
```

!!! warning

    Under construction

!!! warning

	This section is only an implementation guide. The definitive specification of the
	Learn API is given in [Reference](@ref reference).

This guide is intended to be consulted after reading [Anatomy of an Implementation](@ref),
which introduces the main interface objects and terminology.

Although an implementation is defined purely by the methods and traits it implements, many
implementations fall into one (or more) of the following informally understood patterns or
"tasks":

- [Regression](@ref): Supervised learners for continuous targets

- Classification: Supervised learners for categorical targets 

- Clusterering: Algorithms that group data into clusters for classification and
  possibly dimension reduction. May be true learners (generalize to new data) or static.

- Gradient Descent: Including neural networks.

- [Iterative Algorithms](@ref)

- Incremental Algorithms

- [Feature Engineering](@ref): Algorithms for selecting or combining features

- Dimension Reduction: Transformers that learn to reduce feature space dimension

- Missing Value Imputation

- Transformers: Other transformers, such as standardizers, and categorical
  encoders.

- [Static Algorithms](@ref): Algorithms that do not learn, in the sense they must be
  re-executed for each new data set (do not generalize), but which have hyperparameters
  and/or deliver ancillary information about the computation.
  
- [Ensembling](@ref): Algorithms that blend predictions of multiple algorithms

- Time Series Forecasting

- Time Series Classification

- Survival Analysis

- Density Estimation: Algorithms that learn a probability distribution

- Bayesian Algorithms

- Outlier Detection: Supervised, unsupervised, or semi-supervised learners for
  anomaly detection.

- Text Analysis

- Audio Analysis

- Natural Language Processing

- Image Processing

- [Meta-algorithms](@ref)

