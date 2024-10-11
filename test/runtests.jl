using Test

test_files = [
    "tools.jl",
    "traits.jl",
    "clone.jl",
    "fit_update.jl",
    "accessor_functions.jl",
    "predict_transform.jl",
    "patterns/regression.jl",
    "patterns/static_algorithms.jl",
    "patterns/ensembling.jl",
    "patterns/incremental_algorithms.jl",
    "patterns/regression.jl",
    "patterns/static_algorithms.jl",
    "integration/ensembling.jl",
]

files = isempty(ARGS) ? test_files : ARGS

for file in files
    quote
        @testset $file begin
            include($file)
        end
    end |> eval
end
