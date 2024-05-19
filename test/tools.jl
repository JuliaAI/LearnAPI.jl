using LearnAPI
using Test

module Fruit
using LearnAPI

struct RedApple{T}
    x::T
end

@trait(
    RedApple,
    is_pure_julia = true,
    pkg_name = "Fruity",
)

end

import .Fruit


## HELPERS

@testset "snakecase" begin
    @test LearnAPI.snakecase("AnthonyBlaomsPetElk") ==
        "anthony_blaoms_pet_elk"
    @test LearnAPI.snakecase("TheLASERBeam", delim=' ') ==
        "the laser beam"
    @test LearnAPI.snakecase(:TheLASERBeam) == :the_laser_beam
end

@testset "@trait" begin
    @test LearnAPI.is_pure_julia(Fruit.RedApple(1))
    @test LearnAPI.pkg_name(Fruit.RedApple(1)) == "Fruity"
end

true
