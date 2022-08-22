# Model Traits

| trait | fallback value | requires | required by |
|:------|:---------|:---------|:---------------|
| [`LearnAPI.ismodel`](@ref) | `false` | one of: `predict`/`predict_joint`/`transform` | all models |
| [`LearnAPI.implemented_methods`](@ref) | `Symbol[]` | | all models |
| [`LearnAPI.is_supervised`](@ref) | `false` | [`LearnAPI.predict`](@ref) or [`LearnAPI.predict_joint`](@ref) | [`LearnAPI.predict_joint`](@ref) |
| [`LearnAPI.paradigm`](@ref) | `:unknown` | relevant operations | [`LearnAPI.predict`](@ref), [`MLJInterface.predict_joint`](@ref) †|
| [`MLInteface.joint_prediction_type`](@ref) | `:unknown` | [`LearnAPI.predict_joint`](@ref) | [`LearnAPI.predict_joint`](@ref) |

† If additionally `is_supervised(model) == true`. 
