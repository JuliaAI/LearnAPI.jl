module LearnAPI

using Statistics

include("tools.jl")
include("algorithms.jl")
include("fit_update_ingest.jl")
include("operations.jl")
include("accessor_functions.jl")
include("data_interface.jl")
include("algorithm_traits.jl")

export @trait

end
