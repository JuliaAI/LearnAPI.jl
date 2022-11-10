# Accessor Functions 

> **Summary.** While byproducts of training are ordinarily recorded in the `report`
> component of the output of `fit`/`update!`/`ingest!`, some families of models report an
> itme that is likely shared by multiple model types, and it is useful to have common
> interface for accessing these directly. Training losses and feature importances are two
> examples.

```@docs
LearnAPI.feature_importances
LearnAPI.training_labels
LearnAPI.training_losses
LearnAPI.training_scores
```

