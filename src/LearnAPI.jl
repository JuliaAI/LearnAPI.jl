module LearnAPI

import InteractiveUtils.subtypes

include("tools.jl")
include("types.jl")
include("predict_transform.jl")
include("fit.jl")
include("minimize.jl")
include("target_weights_features.jl")
include("obs.jl")
include("accessor_functions.jl")
include("traits.jl")

export @trait
export fit, update, update_observations, update_features
export predict, transform, inverse_transform, minimize, obs

for name in Symbol.(CONCRETE_TARGET_PROXY_TYPES_SYMBOLS)
    @eval export $name
end

end # module
