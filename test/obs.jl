using Test
using LearnAPI

@testset "`obs` fallback" begin
    @test obs("some learner", 42) == 42
end

true
