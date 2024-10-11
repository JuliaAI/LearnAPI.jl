using Test

test_files = [
    "tools.jl",
    "traits.jl",
    "clone.jl",
    "integration/regression.jl",
    "integration/static_algorithms.jl",
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
