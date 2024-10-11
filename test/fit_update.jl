using Test
using LearnAPI

struct Gander end

LearnAPI.update(::Gander, data) = sum(data)
LearnAPI.update_features(::Gander, data) = prod(data)

@testset "update, update_features slurping" begin
    @test update(Gander(), 2, 3, 4) == 9
    @test update_features(Gander(), 2, 3, 4) == 24
end

true
