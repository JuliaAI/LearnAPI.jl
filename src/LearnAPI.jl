module LearnAPI

include("types.jl")
include("tools.jl")
include("predict_transform.jl")
include("fit_update.jl")
include("target_weights_features.jl")
include("obs.jl")
include("accessor_functions.jl")
include("traits.jl")
include("clone.jl")

export @trait, @functions, clone
export fit, update, update_observations, update_features
export predict, transform, inverse_transform, obs

for name in CONCRETE_TARGET_PROXY_SYMBOLS
    @eval export $name
end

end # module
