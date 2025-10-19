# Testing an Implementation

Testing is provided by the LearnTestAPI.jl package.

Testing is provided by the LearnTestAPI.jl package documented below.

## Quick start

```@docs
LearnTestAPI
```

!!! warning

	New releases of LearnTestAPI.jl may add tests to `@testapi`, and
	this may result in new failures in client package test suites, because
	of previously undetected broken contracts. Adding a test to `@testapi`
	is not considered a breaking change
	to LearnTestAPI, unless it supports a breaking change to LearnAPI.jl.


## The @testapi macro

```@docs
LearnTestAPI.@testapi
```

## Learners for testing

LearnTestAPI.jl provides some simple, tested, LearnAPI.jl implementations, which may be
useful for testing learner wrappers and meta-algorithms.

```@docs
LearnTestAPI.Ridge
LearnTestAPI.BabyRidge
LearnTestAPI.ConstantClassifier
LearnTestAPI.TruncatedSVD
LearnTestAPI.Selector
LearnTestAPI.FancySelector
LearnTestAPI.NormalEstimator
LearnTestAPI.Ensemble
LearnTestAPI.StumpRegressor
```

## Private methods

For LearnTestAPI.jl developers only, and subject to breaking changes at any time:

```@docs
LearnTestAPI.@logged_testset
LearnTestAPI.@nearly
LearnTestAPI.isnear
LearnTestAPI.learner_get
LearnTestAPI.model_get
LearnTestAPI.verb
LearnTestAPI.filter_out_verbosity
```
