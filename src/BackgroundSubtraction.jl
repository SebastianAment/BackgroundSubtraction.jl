module BackgroundSubtraction
using LinearAlgebra
using LinearAlgebraExtensions
using LinearAlgebraExtensions: LowRank, pals!, Projection, projection
using Kernel
using Statistics

export mcbl
export kronecker_mcbl

include("background.jl")

end
