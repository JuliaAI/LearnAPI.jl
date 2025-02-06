using Test
using LearnAPI

struct Avocado end

@test LearnAPI.target(Avocado(), (1, 2, 3)) == 3
@test isnothing(LearnAPI.weights(Avocado(), "salsa"))
@test LearnAPI.features(Avocado(), "salsa") == "salsa"
@test LearnAPI.features(Avocado(), (:X, :y)) == :X

true
