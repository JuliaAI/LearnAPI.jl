using Documenter
using LearnAPI
using ScientificTypesBase

const  REPO = Remotes.GitHub("JuliaAI", "LearnAPI.jl")

makedocs(
    modules=[LearnAPI,],
    format=Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        collapselevel = 1,
    ),
    pages=[
        "Home" => "index.md",
        "Anatomy of an Implementation" => "anatomy_of_an_implementation.md",
        "Reference" => [
            "Overview" => "reference.md",
            "fit/update" => "fit_update.md",
            "predict/transform" => "predict_transform.md",
            "Kinds of Target Proxy" => "kinds_of_target_proxy.md",
            "target/weights/features" => "target_weights_features.md",
            "obs" => "obs.md",
            "Accessor Functions" => "accessor_functions.md",
            "Learner Traits" => "traits.md",
        ],
        "Common Implementation Patterns" => "common_implementation_patterns.md",
        "Testing an Implementation" => "testing_an_implementation.md",
    ],
    sitename="LearnAPI.jl",
    warnonly = [:cross_references, :missing_docs],
    repo = Remotes.GitHub("JuliaAI", "LearnAPI.jl"),
)

deploydocs(
    devbranch="dev",
    push_preview=false,
    repo="github.com/JuliaAI/LearnAPI.jl.git",
)
