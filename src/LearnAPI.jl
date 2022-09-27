module LearnAPI

using Statistics

include("tools.jl")
include("models.jl")
include("fit_update_ingest.jl")
include("operations.jl")
include("model_traits.jl")

export @trait

end
