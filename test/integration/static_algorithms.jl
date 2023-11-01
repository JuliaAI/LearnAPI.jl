using LearnAPI
using LinearAlgebra
using Tables
import MLUtils
import DataFrames


# # TRANSFORMER TO SELECT SOME FEATURES (COLUMNS) OF A TABLE

# See later for a variation that stores the names of rejected features in the model
# object, for inspection by an accessor function.

struct Selector
    names::Vector{Symbol}
end
Selector(; names=Symbol[]) =  Selector(names)

LearnAPI.obsfit(algorithm::Selector, obsdata, verbosity) = algorithm
LearnAPI.algorithm(model) = model # i.e., the algorithm

function LearnAPI.obstransform(algorithm::Selector, obsdata)
    X = only(obsdata)
    table = Tables.columntable(X)
    names = Tables.columnnames(table)
    filtered_names = filter(in(algorithm.names), names)
    filtered_columns = (Tables.getcolumn(table, name) for name in filtered_names)
    filtered_table = NamedTuple{filtered_names}((filtered_columns...,))
    return Tables.materializer(X)(filtered_table)
end

@trait Selector functions = (
    fit,
    obsfit,
    minimize,
    transform,
    obstransform,
    obs,
    Learn.algorithm,
)

@testset "test a static transformer" begin
    algorithm = Selector(names=[:x, :w])
    X = DataFrames.DataFrame(rand(3, 4), [:x, :y, :z, :w])
    model = fit(algorithm) # no data arguments!
    @test model == algorithm
    @test transform(model, X) ==
        DataFrames.DataFrame(Tables.matrix(X)[:,[1,4]], [:x, :w])
end


# # FEATURE SELECTOR THAT REPORTS BYPRODUCTS OF SELECTION PROCESS

# This a variation of `Selector` above that stores the names of rejected features in the
# model object, for inspection by an accessor function called `rejected`.

struct Selector2
    names::Vector{Symbol}
end
Selector2(; names=Symbol[]) =  Selector2(names)

mutable struct Selector2Fit
    algorithm::Selector2
    rejected::Vector{Symbol}
    Selector2Fit(algorithm) = new(algorithm)
end
LearnAPI.algorithm(model::Selector2Fit) = model.algorithm
rejected(model::Selector2Fit) = model.rejected

# Here `obsdata=()` and we are just wrapping `algorithm` with a place-holder for
# the `rejected` feature names.
LearnAPI.obsfit(algorithm::Selector2, obsdata, verbosity) = Selector2Fit(algorithm)

# output the filtered table and add `rejected` field to model (mutatated)
function LearnAPI.obstransform(model::Selector2Fit, obsdata)
    X = only(obsdata)
    table = Tables.columntable(X)
    names = Tables.columnnames(table)
    keep = LearnAPI.algorithm(model).names
    filtered_names = filter(in(keep), names)
    model.rejected = setdiff(names, filtered_names)
    filtered_columns = (Tables.getcolumn(table, name) for name in filtered_names)
    filtered_table = NamedTuple{filtered_names}((filtered_columns...,))
    return Tables.materializer(X)(filtered_table)
end

@trait(
    Selector2,
    predict_or_transform_mutates = true,
    functions = (
        fit,
        obsfit,
        minimize,
        transform,
        obstransform,
        obs,
        Learn.algorithm,
        :(MyPkg.rejected), # accessor function not owned by LearnAPI.jl
    )
)

@testset "test a variation that reports byproducts" begin
    algorithm = Selector2(names=[:x, :w])
    X = DataFrames.DataFrame(rand(3, 4), [:x, :y, :z, :w])
    model = fit(algorithm) # no data arguments!
    @test !isdefined(model, :reject)
    @test LearnAPI.algorithm(model) == algorithm
    filtered =  DataFrames.DataFrame(Tables.matrix(X)[:,[1,4]], [:x, :w])
    @test transform(model, X) == filtered
    @test rejected(model) == [:y, :z]
end

true
