using BackgroundSubtraction
using CSV
using DataFrames

# read in data, should be one directory above "BackgroundSubtraction" directory
dir = "BackgroundSubtraction/data/mcbl_raman_dataset/"
read(name) = Matrix(CSV.read(dir * name * ".csv", DataFrame))
wavenumber, data = vec(read("wavenumber")), read("mcbl_raman_dataset")

# sample_id, x, y, composition (6d), intensity
data = data' # transposing to put each spectrogram in a matrix COLUMN
sample_id = data[1, :]
x, y = data[2, :], data[3, :]  # spatial coordinates on waver
composition = data[4:9, :]  # elemental composition (Mn, Fe, Ni, Cu, Co, Zn)
intensity = data[10:end, :]  # spectrographic intensity as a function of wavenumber

# sanity checks
n = size(composition, 2)
@assert(sum(composition, dims = 1) ≈ ones(1, num_spectrograms))  # composition sums up to one
@assert(length(wavenumber) == size(intensity, 1))  # wavenumbers and intensities match

# normalize variables
wavenumber /= maximum(wavenumber)
intensity /= maximum(intensity)

# subsample variables for faster inference, can get rid of this step without
# change in background estimation quality.
m, n = size(intensity)
if true
    subsampling = 1:2:m
    wavenumber = wavenumber[subsampling]
    intensity = intensity[subsampling, :]
    m, n  = size(intensity)
end

# lengthscale for wavenumber as a fraction of the total domain
l_λ = 0.05

k = 5  # number of background components
# two non-default modifications to the mcbl call:
# 1) bumping maxiter to 64 to ensure convergence.
# 2) reducing minres to 1e-3 to ensure no pre-mature termination,
# since noise level of normalized data is small.
background = BackgroundSubtraction.mcbl(
                intensity,
                k,
                wavenumber,
                l_λ,  # lenth scale with respect to wavelength
                minres = 1e-3,  # convergence criterion, stop when std of residual hits minres
                maxiter = 64,  # maximum number of iterations to run
            )

using Plots
# plotting overview of all spectrograms to judge overall result:
# looks good! a few notable signals towards the right end of the domain,
# that would be particularly difficult to catch otherwise.
plot(wavenumber, intensity - background, legend=false)
gui()
readline()

challenging_examples = [47, 206, 212, 257, 350, 356, 704, 799, 959]
# 47, 212, 257,
# 206 all signal,
# 350, 356 really complex, with many small peaks on top of large substrate peak
# 704, 799 are difficult with large peak intensities
# 959 has a peak on top of a substrate peak at the endpoint of the spectrogram,
# this would be impossible to infer with a method that does not accumulate
# information across the entire dataset.
for i in challenging_examples
    print(i)
    plot(wavenumber, intensity[:, i], legend=false)
    plot!(wavenumber, background[:, i])
    plot!(wavenumber, intensity[:, i] - background[:, i])
    gui()
    readline()
end
