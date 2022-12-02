module Fruit

struct RedApple{T}
    x::T
end

end

import .Fruit


## HELPERS

@testset "typename" begin
    @test StatisticalTraits.typename(Fruit.RedApple) == :RedApple
    @test StatisticalTraits.typename(Fruit.RedApple{Int}) == :RedApple
    @test StatisticalTraits.typename(Nothing) == :Nothing
    @test StatisticalTraits.typename(UnionAll) == :UnionAll
    @test StatisticalTraits.typename(Union{Char,Int}) ==
        Symbol("Union{Char, Int64}")
    T = SparseArrays.sparse([1,2], [1,3], [0.5, 0.6]) |> typeof
    @test StatisticalTraits.typename(T) == :SparseMatrixCSC
end

@testset "snakecase" begin
    @test StatisticalTraits.snakecase("AnthonyBlaomsPetElk") ==
        "anthony_blaoms_pet_elk"
    @test StatisticalTraits.snakecase("TheLASERBeam", delim=' ') ==
        "the laser beam"
    @test StatisticalTraits.snakecase(:TheLASERBeam) == :the_laser_beam
end

true
