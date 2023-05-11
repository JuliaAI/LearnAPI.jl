module LearnAPI

using Statistics
import InteractiveUtils.subtypes

include("tools.jl")
include("algorithms.jl")
include("operations.jl")
include("fit_update_ingest.jl")
include("accessor_functions.jl")
include("data_interface.jl")
include("algorithm_traits.jl")

export @trait

end
