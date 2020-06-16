module TestBackground
using Test
using BackgroundSubtraction: mcbl
using LinearAlgebra
function synthetic_data(n = 128, m = 16)
    f(x) = sin(2π*x) + 1
    x = range(0, stop = 1, length = n)
    fx = f.(x)
    A = repeat(fx, 1, m)

    # add noise
    σ = 1e-2
    @. A += σ * randn()
    # add positive outliers (i.e. peaks)
    npeaks = n*m ÷ 3
    outind = rand(eachindex(A), npeaks)
    @. A[outind] += exp(randn())
    return A, x, fx
end

@testset "background" begin
    n, m = 128, 16
    A, x, fx = synthetic_data(n, m)
    l = .2
    tol = 1e-1
    background = mcbl(A, 1, x, l)
    @test size(background) == size(A)
    @test all(≥(0), background)
    @test maximum(abs, background[:, 1]-fx) < tol
    i = 1

    doplot = true
    if doplot
        using Plots
        plotly()
        plot(x, A[:,i], label = "data")
        plot!(x, background[:, i], label = "background")
        plot!(x, fx, label = "inferred background")
        gui()
    end

    # testing with only vector input: less powerful, and needs stronger l regularization
    tol = 3e-1
    background = mcbl(A[:,i], x, l)
    @test background isa Vector
    @test length(background) == length(A[:, i])

end

end # TestBackground
