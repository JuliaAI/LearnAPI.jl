module LearnAPI

using Statistics
using InteractiveUtils

include("tools.jl")
include("models.jl")
include("fit_update_ingest.jl")
include("operations.jl")
include("accessor_functions.jl")
include("model_traits.jl")

export @trait

end
