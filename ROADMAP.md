# Road map

- [ ] Mock up a challenging `update` use-case: controlling an iterative algorithm that
      wants, for efficiency, to internally compute the out-of-sample predictions that will
      be for used to *externally* determined early stopping cc: @jeremiedb

- [ ] Get code coverage to 100% (see next item)

- [ ] Add to this repo or a utility repo methods to test a valid implementation of
	  LearnAPI.jl
	  
- [ ] Flush out "Common Implementation Patterns". The current plan is to mock up example
	  implementations, and add them as LearnAPI.jl tests, with links to the test file from
	  "Common Implementation Patterns". As real-world implementations roll out, we could
	  increasingly point to those instead, to conserve effort
	  - [x] regression
	  - [ ] classification
      - [ ] clustering
	  - [ ] gradient descent
	  - [ ] iterative algorithms
	  - [ ] incremental algorithms
	  - [ ] dimension reduction
	  - [x] feature engineering
	  - [x] static algorithms
	  - [ ] missing value imputation
	  - [ ] transformers
	  - [ ] ensemble algorithms
	  - [ ] time series forecasting
	  - [ ] time series classification
	  - [ ] survival analysis
	  - [ ] density estimation
	  - [ ] Bayesian algorithms
	  - [ ] outlier detection
	  - [ ] collaborative filtering
	  - [ ] text analysis
	  - [ ] audio analysis
	  - [ ] natural language processing
	  - [ ] image processing
	  - [ ] meta-algorithms

- [ ] In a utility package provide:
    - [ ] Method to clone an algorithm with user-specified property(hyperparameter)
          changes, as in `LearnAPI.clone(algorithm, p1=value1, p22=value2, ...)` (since
          `algorithm` can have any type, can't really overload `Base.replace` without
          piracy). This will be needed in tuning meta-algorithms. Or should this be in
          LearnAPI.jl proper, to expose it to all users?
	- [ ] Methods to facilitate common-use case data interfaces: support simultaneously
          `fit` data of the form `data = (X, y)` where `X` is table *or* matrix, and
          `data` a table with target specified by hyperparameter; here `obs` will return a
          thin wrapping of the matrix of `X`, the target `y`, and the names of all
          fields. We can have options to make `X` a concrete array or an adjoint,
          depending on what is more efficient for the algorithm. 
