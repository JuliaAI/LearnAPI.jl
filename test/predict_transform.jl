using Test
using LearnAPI

struct Goose end

LearnAPI.fit(algorithm::Goose) = Ref(algorithm)
LearnAPI.algorithm(::Base.RefValue{Goose}) = Goose()
LearnAPI.predict(::Base.RefValue{Goose}, ::Point, data) = sum(data)
LearnAPI.transform(::Base.RefValue{Goose}, data) = prod(data)
@trait Goose kinds_of_proxy = (Point(),)

@testset "predict and transform argument slurping" begin
    model = fit(Goose())
    @test predict(model, Point(), 2, 3, 4) == 9
    @test predict(model, 2, 3, 4) == 9
    @test transform(model, 2, 3, 4) == 24
end

true
