using Test
using LearnAPI

struct Avacado end

@test isnothing(LearnAPI.target(Avacado(), "salsa"))
@test isnothing(LearnAPI.weights(Avacado(), "salsa"))
@test LearnAPI.features(Avacado(), "salsa") == "salsa"
@test LearnAPI.features(Avacado(), (:X, :y)) == :X

true
