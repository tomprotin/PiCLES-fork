module Models

export WaveGrowthModels1D, WaveGrowthModels2D, ParametricModels, GeometricalOpticsModels, reset_boundary!

include("WaveGrowthModels1D.jl")
using .WaveGrowthModels1D
include("WaveGrowthModels2D.jl")
using .WaveGrowthModels2D
include("ParametricModels.jl")
using .ParametricModels
include("GeometricalOpticsModels.jl")
using .GeometricalOpticsModels


end