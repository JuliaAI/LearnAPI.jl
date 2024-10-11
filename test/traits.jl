using Test
using LearnAPI

# A MINIMUM IMPLEMENTATION OF AN ALGORITHM

# does nothing useful
struct SmallAlgorithm end
LearnAPI.fit(algorithm::SmallAlgorithm, data; verbosity=1) = algorithm
LearnAPI.algorithm(model::SmallAlgorithm) = model
@trait(
    SmallAlgorithm,
    constructor = SmallAlgorithm,
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.algorithm),
        :(LearnAPI.strip),
        :(LearnAPI.obs),
        :(LearnAPI.features),
    ),
)
######## END OF IMPLEMENTATION ##################

# ZERO ARGUMENT METHODS

@test :(LearnAPI.fit) in LearnAPI.functions()
@test Point in LearnAPI.kinds_of_proxy()
@test "regression" in LearnAPI.tags()

# OVERLOADABLE TRAITS

small = SmallAlgorithm()
@test LearnAPI.constructor(small) == SmallAlgorithm
@test :(LearnAPI.algorithm) in LearnAPI.functions(small)
@test isempty(LearnAPI.kinds_of_proxy(small))
@test isempty(LearnAPI.tags(small))
@test !LearnAPI.is_pure_julia(small)
@test LearnAPI.pkg_name(small) == "unknown"
@test LearnAPI.pkg_license(small) == "unknown"
@test LearnAPI.doc_url(small) == "unknown"
@test LearnAPI.load_path(small) == "unknown"
@test !LearnAPI.is_composite(small)
@test LearnAPI.human_name(small) == "small algorithm"
@test isnothing(LearnAPI.iteration_parameter(small))
@test LearnAPI.data_interface(small) == LearnAPI.RandomAccess()
@test !(6 isa LearnAPI.fit_observation_scitype(small))
@test 6 isa LearnAPI.target_observation_scitype(small)

# DERIVED TRAITS

@test LearnAPI.is_algorithm(small)
@test !LearnAPI.target(small)
@test !LearnAPI.weights(small)

module FruitSalad
import LearnAPI

struct RedApple{T}
    x::T
end

LearnAPI.constructor(::RedApple) = RedApple

end

import .FruitSalad

@testset "name" begin
    @test LearnAPI.name(FruitSalad.RedApple(1)) == "RedApple"
end
