import numpy as np
import matplotlib.pyplot as plt
import os

#Order is important!!
#set the number of cores in the julia environment
from multiprocessing import cpu_count
os.environ["JULIA_NUM_THREADS"] = str(cpu_count())
from julia import Julia

#For windows and linux, there is the bug that requires the system image be passed into Julia
Julia(sysimage="./sys.so")
from julia import Main
#using the julia syntax, you add the BackgroundSubtraction.jl code to the julia Main and can now import the julia functions.
Main.include('/home/duncan/BackgroundSubtraction.jl/src/BackgroundSubtraction.jl')
from julia.Main import BackgroundSubtraction
mcbl = BackgroundSubtraction.mcbl


if __name__ == '__main__':
    #1D example
    #load 1D data
    Y = np.load('sample_1D_Data.npy')
    X = np.load('sample_x_Data.npy')
    #The MCBL code applied to 1D data (i.e. Intensity vs. Q in x-ray diffraction)
    # - the 1D data                       type:     (1 x n) float64
    # - the x array                       type:     Array(1 x n) float64
    # - length scale                      type:     float

    Y = Y
    X = X.astype('float64')
    l_1 = 8.                                          # in units of X

    bkg = mcbl(Y, X, l_1)

    fig,ax = plt.subplots(dpi=150)
    ax.plot(X, Y,   label='raw')
    ax.plot(X, bkg, c='gray', linestyle='--', label='background')
    ax.plot(X, Y-bkg, label='removed')

    ax.set(
        xlabel="Q in nm$^{-1}",
        ylabel="Intensity (a.u)",
        title="1D example"
    )
    ax.legend()
    plt.show()
    



    #2D example
    #2D data of Y values and the corresponding X vector
    Y = np.load('sample_2D_Data.npy')
    X = np.load('sample_x_Data.npy')
    
    #The MCBL code applied to 2D data (in this case intensity of x-ray diffraction patterns as a function of position)
    #The inputs in order are
    # - the 2D data                       type:     (m x n) float64
    # - the x array                       type:     Array(1 x n) float64
    # - length scale of Y dim 1           type:     float
    # - length scale of Y dim 2           type:     Array(m x 1) float64
    # - nIterations                       type:     float
    bkg = mcbl(Y,X.astype('float64'),5.,10*np.arange(Y.shape[1],dtype='float64'),1.)
    
    

    #plotting to show the results
    fig,ax = plt.subplots(1,2,figsize=(9,9))
    fig.suptitle("2D example")
    for idx, y in enumerate(Y.T):
        ax[0].plot(X,y+idx*1)
    
    for idx, b in enumerate(bkg.T):
        ax[0].plot(X,b+idx*1,c='k',linestyle='--')
    
    for idx, y in enumerate(Y.T):
        ax[1].plot(X, y - bkg.T[idx] + idx)
    
    ax[0].set(
        xlabel = "x data",
        ylabel = "y data",
        title = "signals and backgrounds"
        )

    ax[1].set(
        xlabel = "x data",
        ylabel = "y data",
        title = "background subtracted"
        )
    plt.show() 
