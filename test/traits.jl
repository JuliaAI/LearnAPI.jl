module FruitSalad
using LearnAPI

struct RedApple{T}
    x::T
end

LearnAPI.constructor(::RedApple) = RedApple

end

import .FruitSalad

@testset "name" begin
    @test LearnAPI.name(FruitSalad.RedApple(1)) == "RedApple"
end
