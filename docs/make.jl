using Documenter
using LearnAPI
using ScientificTypesBase

const REPO="github.com/JuliaAI/LearnAPI.jl"

makedocs(;
    modules=[LearnAPI,],
    format=Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages=[
        "Overview" => "index.md",
        "Anatomy of an Implementation" => "anatomy_of_an_implementation.md",
        "Reference" => "reference.md",
        "Fit, update and ingest" => "fit_update_and_ingest.md",
        "Predict and other operations" => "operations.md",
        "Model Traits" => "model_traits.md",
        "Common Implementation Patterns" => "common_implementation_patterns.md",
        "Testing an Implementation" => "testing_an_implementation.md",
    ],
    repo="https://$REPO/blob/{commit}{path}#L{line}",
    sitename="LearnAPI.jl"
)

deploydocs(
    ; repo=REPO,
    devbranch="dev",
    push_preview=false,
)

