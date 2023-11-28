module LearnAPI

import InteractiveUtils.subtypes

include("tools.jl")
include("types.jl")
include("predict_transform.jl")
include("fit.jl")
include("minimize.jl")
include("obs.jl")
include("accessor_functions.jl")
include("traits.jl")

export @trait
export fit, predict, transform, inverse_transform, fit_transform, minimize
export obs, obsfit, obspredict, obstransform

for name in Symbol.(CONCRETE_TARGET_PROXY_TYPES_SYMBOLS)
    @eval export $name
end

end # module
