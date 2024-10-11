using Test

test_files = [
    "tools.jl",
    "traits.jl",
    "clone.jl",
    "predict_transform.jl",
    "patterns/regression.jl",
    "patterns/static_algorithms.jl",
    "patterns/ensembling.jl",
    "patterns/incremental_algorithms.jl",
]

files = isempty(ARGS) ? test_files : ARGS

for file in files
    quote
        @testset $file begin
            include($file)
        end
    end |> eval
end
