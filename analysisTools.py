import numpy as np
import xarray as xr
from scipy import interpolate
from finitediff import get_weights
sectoyear=365.25 * 24 * 60 * 60
import time

def ntsteps_to_seconds(n):
    """Convert a number of Calypso time steps to dimensional seconds"""
    t_step = 1e-2 # tstep diffusionless = tstep_calypso / E_calypso = 1e-7/1e-5 = 1e-2
    Omega_adjusted = 4.405286343612335e-08#4.4e-8
    return n * t_step / Omega_adjusted

def seconds_to_ntsteps(s):
    """Convert dimensional seconds to a number of Calypso time steps"""
    t_step = 1e-2 # tstep diffusionless = tstep_calypso / E_calypso = 1e-7/1e-5 = 1e-2
    Omega_adjusted = 4.405286343612335e-08#4.4e-8
    return s/(t_step / Omega_adjusted)

def nonzero_frequencies():
    """Get the nonzero frequencies (in s^-1) of a Fourier transformed array (where the original array was on a time grid of 800 times with dt=25 Calypso time steps)"""
    freqs = np.fft.fftfreq(800,ntsteps_to_seconds(25))[1:]
    return xr.DataArray(freqs,coords={'frequency':freqs},dims = ['frequency'],attrs={'unit':'s^-1'})

def freqindex_to_period(idx):
    """Goes from the index of a Fourier transformed array (where the original array was on a time grid of 800 times with dt=25 Calypso time steps) to a dimensional period in seconds """
    return 1/np.array(nonzero_frequencies())[idx]

def period_to_freqindex(period):
    """Goes from a dimensional period in seconds to the index of a Fourier transformed array (where the original array was on a time grid of 800 times with dt=25 Calypso time steps)"""
    return np.argmin((1/np.array(nonzero_frequencies())-period)**2)

def put_at(inds, axis=-1, slc=(slice(None),)): 
    return (axis<0)*(Ellipsis,) + axis*slc + (inds,) + (-1-axis)*slc 

def fft_halfrange_sin(x,axis=-1):
    """ Computes the sine transform of an array on half range (look up 'half range fourier series'). Works the same fft, except it returns one array, 
    corresponding to the terms of the half range sine transform
    such that x_m = sum(k=1 to N-1) s_k sin(m*k*pi/N)
    args :
        - x : numpy.ndarray, array to be transformed
        - axis : axis along which to compute the transform
    returns :
        - numpy.ndarray of the same shape as x
    """
    N=x.shape[axis]
    x2 = np.concatenate([x,-np.flip(x,axis=axis)],axis=axis)
    fft = np.fft.fft(x2,axis=axis)
    sineterms = 1j*(np.take(fft,range(1,N+1),axis=axis)-np.take(fft,range(2*N-1,N-1,-1),axis=axis))
    sineterms[put_at(N-1,axis=axis)] = 1j*np.take(fft,N,axis=axis)
    return sineterms/2/N

def fft_halfrange_cos(x,axis=-1):
    """ Computes the cosine transform of an array on half range (look up 'half range fourier series'). Works the same as np.fft.fft, except it returns one array, 
    corresponding to the terms of the half range cosine transform
    such that x_m = sum(k=0 to N-1) c_k cos(m*k*pi/N)
    Note that the length of the returned array along axis is one more than the input
    args :
        - x : numpy.ndarray, array to be transformed
        - axis : axis along which to compute the transform
    returns :
        - numpy.ndarray of the same shape as x except along the axis, where its length is incremented by 1
    """
    N=x.shape[axis]
    x2 = np.concatenate([x,np.flip(x,axis=axis)],axis=axis)
    fft = np.fft.fft(x2,axis=axis)
    cosineterms = (np.take(fft,range(1,N+1),axis=axis)+np.take(fft,range(2*N-1,N-1,-1),axis=axis))
    cosineterms[put_at(N-1,axis=axis)] = np.take(fft,N,axis=axis)
    cosineterms=np.concatenate([np.take(fft,[0.],axis=axis),cosineterms],axis=axis)
    return cosineterms/2/N

def make_D_fornberg(y,m,npoints=5):
    """Returns a differentiation matrix for the mth derivative on a nonuniform grid y, using a npoints-point stencil (uses Fornberg's algorithm)
    """
    N=len(y)
    assert N>=npoints
    D=np.zeros((N,N))
    for i in range(npoints//2):
        D[ i,:npoints] = get_weights(y[:npoints],y[i],-1,m)[:,m]
        D[-i-1,-npoints:] = get_weights(y[-npoints:],y[-i-1],-1,m)[:,m] 
    for i in range(npoints//2,N-npoints//2):
        D[i,i-npoints//2:i+npoints//2+1] = get_weights(y[i-npoints//2:i+npoints//2+1],y[i],-1,m)[:,m]   
    return D

def transform_forcing(forcing,transformtype):
    """Given a forcing on a (time, radius, y) grid, take its half range Fourier transform in radius
    (either sine or cosine) and its temporal Fourier transform
    args :
        - forcing : numpy.ndarray, forcing to be transformed. Should be 3-dimensional
        - transformtype : str, either 'cos' or 'sin', specifies the type of half range Fourier transform that should be computed
    returns :
        - numpy.ndarray, transformed forcing
    """
    #Focus on stratified layer
    forcing_layer = forcing[:,np.where(forcing.radius>1.475)[0][0]:]
    
    #Compute radial grid on which to interpolate before taking radial Fourier transform
    r_cheb = np.array(forcing_layer.radius)
    r_lin = np.linspace(r_cheb[0],r_cheb[-1],len(r_cheb))
    Nz=len(r_cheb)
    Nt=len(forcing_layer.t_step)
    
    #Here we subtract the affine part of the forcing, that doesn't contribute because what ultimately forces the wave is the second derivative (or third). This helps have smooth Fourier transforms in the subsequent decomposition
    forcing_layer_periodic = forcing_layer - forcing_layer.isel(radius_ID=-1) - (forcing_layer.radius-forcing_layer.radius[-1])*(forcing_layer.isel(radius_ID=0)-forcing_layer.isel(radius_ID=-1))/(forcing_layer.radius[0]-forcing_layer.radius[-1])

    #Interpolate on constant r grid & perform the half range fft
    f = interpolate.interp1d(r_cheb, np.array(forcing_layer_periodic),axis=1)
    forcing_layer_rinterp = f(r_lin)
    if transformtype == "sin":
        forcing_layer_hft=fft_halfrange_sin(np.array(forcing_layer_rinterp),axis=1)
    elif transformtype == "cos":
        forcing_layer_hft=fft_halfrange_cos(np.array(forcing_layer_rinterp),axis=1)[:,1:]
    else :
        raise ValueError("transformtype must be one of ('sin','cos')")
    
    # Perform temporal fft
    forcing_layer_hft_tft = np.fft.ifft(forcing_layer_hft,axis=0)[1:] #shape Nt,Nz,Ny #ifft because we define the temporal transform with a negative sign in the exponent; also excluded the zero frequency component

    #convert to xarray
    frequencies = nonzero_frequencies()
    return xr.DataArray(forcing_layer_hft_tft,coords={'frequency':frequencies,'radial_order':np.arange(1,forcing_layer_hft_tft.shape[1]+1),'y':forcing_layer.y}, dims=['frequency','radial_order','y'],attrs=forcing.attrs)

def split_NH_SH(forcing):
    forcing_NH = forcing.sel(y=slice(0,1))
    forcing_SH = forcing.sel(y=slice(-1,0))
    forcing_SH = forcing_SH.reindex(y=forcing_SH.y[::-1])
    forcing_SH.coords['y'] = forcing_NH.y
    forcing_NH.coords['latitude'] = np.arcsin(forcing_NH.y)*180.0/np.pi
    forcing_SH.coords['latitude'] = np.arcsin(forcing_SH.y)*180.0/np.pi
    return forcing_NH,forcing_SH
