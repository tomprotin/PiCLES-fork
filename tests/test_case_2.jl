ENV["JULIA_INCREMENTAL_COMPILE"]=true
using Pkg, Dates
println(now())
Pkg.activate(".")

n_batch = 10

for i in 1:n_batch
    include("test_case_2_batch"*string(i)*".jl")
end

include("batch_join.jl")
println(now())
