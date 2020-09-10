module TestBackground
using Test
using BackgroundSubtraction
using LinearAlgebra
function synthetic_data(n = 128, m = 16)
    f(x) = sin(2π*x) + 1.1
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

function synthetic_images(nx = 128, ny = 128, m = 16)
    f(x, y) = sin(2π*(x+y)) + 1.1
    x = range(0, stop = 1, length = nx)
    y = range(0, stop = 1, length = ny)
    fxy = f.(x, y')
    A = repeat(fxy, 1, 1, m)
    # add noise
    σ = 1e-2
    @. A += σ * randn()
    # add positive outliers (i.e. peaks)
    npeaks = nx*ny*m ÷ 3
    outind = rand(eachindex(A), npeaks)
    @. A[outind] += .1exp(randn())
    return A, x, y, fxy
end

@testset "background" begin
    n, m = 128, 16
    A, x, fx = synthetic_data(n, m)
    k = 2
    l = .5
    background = mcbl(A, k, x, l)
    @test size(background) == size(A)
    @test all(≥(0), background)
    tol = 5e-2
    @test maximum(abs, background[:, 1]-fx) < tol
    i = 1

    doplot = false
    if doplot
        using Plots
        plotly()
        plot(x, A[:,i], label = "data")
        plot!(x, background[:, i], label = "inferred background")
        plot!(x, fx, label = "background")
        gui()
    end

    # testing with only vector input: less powerful, and needs stronger l regularization
    background = mcbl(A[:,i], x, l, maxiter = 16)
    @test background isa Vector
    @test length(background) == length(A[:, i])
    tol = 5e-2
    @test maximum(abs, background[:, 1]-fx) < tol

    # testing smooth 2D projection
    lc = 1. # composition length scale
    c = randn(2, m) # 2 dimensional composition dimension
    a = A
    background = mcbl(a, x, l, c, lc)
    tol = 5e-2
    @test maximum(abs, background[:, 1]-fx) < tol

    doplot = false
    if doplot
        using Plots
        plotly()
        plot(x, A[:, i], label = "data")
        plot!(x, background[:, i], label = "inferred background")
        plot!(x, fx, label = "background")
        gui()
    end

    # testing 3D mcbl
    nx, ny, m = 128, 128, 16
    k = 1
    lx, ly = .2, .2
    A, x, y, fxy = synthetic_images(nx, ny, m)
    background = mcbl(A, k, x, lx, y, ly)
    tol = 5e-2
    @test maximum(abs, background[:, :, 1]-fxy) < tol

    # testing 3D smooth projection
    background = mcbl(A, x, lx, y, ly, c, lc)
    tol = 5e-2
    @test maximum(abs, background[:, :, 1]-fxy) < tol

    doplot = false
    if doplot
        using Plots
        plotly()
        surface(x, y, A[:, :, i], label = "data")
        gui()
        surface(x, y, background[:, :, i], label = "inferred background")
        gui()
        surface(x, y, fxy[:, :, i], label = "background")
        gui()
    end

    # testing 3D smooth projection
    # update default parameters

end

end # TestBackground
