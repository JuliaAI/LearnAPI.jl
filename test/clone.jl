using Test
using LearnAPI

struct Potato
    x
    y
end

Potato(; x=1, y=2) = Potato(x, y)
LearnAPI.constructor(::Potato) = Potato

@test LearnAPI.clone(Potato()) == Potato()

p = LearnAPI.clone(Potato(), y=20)
@test p.y == 20
@test p.x == 1

q = LearnAPI.clone(Potato(), y=20, x=10)
@test q.y == 20
@test q.x == 10

true
