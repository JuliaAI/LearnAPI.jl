using Test
using LearnAPI

struct Avocado end

@test isnothing(LearnAPI.target(Avocado(), "salsa"))
@test isnothing(LearnAPI.weights(Avocado(), "salsa"))
@test LearnAPI.features(Avocado(), "salsa") == "salsa"
@test LearnAPI.features(Avocado(), (:X, :y)) == :X

true
