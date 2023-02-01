import numpy as np
import xarray as xr
from scipy import interpolate
from time_tools import nonzero_frequencies

def put_at(inds, axis=-1, slc=(slice(None),)): 
    """
    Function to populate the ith axis of an array
    (taken from https://stackoverflow.com/questions/42817508/get-the-i-th-slice-of-the-k-th-dimension-in-a-numpy-array)
    """
    return (axis<0)*(Ellipsis,) + axis*slc + (inds,) + (-1-axis)*slc 

def fft_halfrange_sin(x,axis=-1):
    """ 
    Computes the sine transform of an array on half range (look up 'half range 
    fourier series'). Works the same as np.fft.fft, except that it returns an array 
    corresponding to the terms of the half range sine transform, i.e. such that 
    x_m = sum(k=1 to N-1) s_k sin(m*k*pi/N)
    
    Parameters
    ----------
    x : numpy.ndarray
        Array to be transformed
    axis : int
        Axis along which to compute the transform.
        
    Returns
    -------
    xhat : numpy.ndarray 
        Coefficients of the half range sine transform (same shape as x).
    """
    N=x.shape[axis]
    x2 = np.concatenate([x,-np.flip(x,axis=axis)],axis=axis)
    fft = np.fft.fft(x2,axis=axis)
    sineterms = 1j*(np.take(fft,range(1,N+1),axis=axis)-np.take(fft,range(2*N-1,N-1,-1),axis=axis))
    sineterms[put_at(N-1,axis=axis)] = 1j*np.take(fft,N,axis=axis)
    return sineterms/2/N

def fft_halfrange_cos(x,axis=-1):
    """ 
    Computes the cosine transform of an array on half range (look up 'half range 
    fourier series'). Works the same as np.fft.fft, except that it returns an array 
    corresponding to the terms of the half range cosine transform, i.e. such that 
    x_m = sum(k=0 to N-1) c_k cos(m*k*pi/N)
    
    Parameters
    ----------
    x : numpy.ndarray
        Array to be transformed
    axis : int
        Axis along which to compute the transform.
        
    Returns
    -------
    xhat : numpy.ndarray 
        Coefficients of the half range sine transform (same shape as x, except
        along the axis, where its length is incremented by 1).
    """
    N=x.shape[axis]
    x2 = np.concatenate([x,np.flip(x,axis=axis)],axis=axis)
    fft = np.fft.fft(x2,axis=axis)
    cosineterms = (np.take(fft,range(1,N+1),axis=axis)+np.take(fft,range(2*N-1,N-1,-1),axis=axis))
    cosineterms[put_at(N-1,axis=axis)] = np.take(fft,N,axis=axis)
    cosineterms=np.concatenate([np.take(fft,[0.],axis=axis),cosineterms],axis=axis)
    return cosineterms/2/N

def transform_forcing(forcing,transformtype,radmin = 1.475):
    """
    Given a forcing on a (time, radius, y) grid, take its half range Fourier 
    transform in radius (either sine or cosine) and its temporal Fourier transform.
    
    Parameters
    ----------
    forcing : numpy.ndarray
        Forcing to be transformed. Must be 3-dimensional, dims = (time, radius, y)
    transformtype : str
        Either 'cos' or 'sin', specifies the type of half range Fourier transform 
        that should be computed.
    radmin : float
        Radius defining the base of the stratified layer
        
    Returns
    -------
    forcing_tilda : numpy.ndarray
        Fourier-transformed forcing.
    """
    #Focus on stratified layer
    forcing_layer = forcing[:,np.where(forcing.radius>radmin)[0][0]:]
    
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
    """
    Splits the northern hemisphere and southern hemisphere parts of a 
    variable.
    
    Parameters
    ----------
    forcing : xarray.Dataset
        Variable to be split. Must have a 'y' (=cos(latitude)) dimension
        
    Returns
    -------
    forcing_NH : xarray.Dataset
        Northern hemisphere part.
    forcing_SH : xarray.Dataset
        Southern hemisphere part.
        
    """
    forcing_NH = forcing.sel(y=slice(0,1))
    forcing_SH = forcing.sel(y=slice(-1,0))
    forcing_SH = forcing_SH.reindex(y=forcing_SH.y[::-1])
    forcing_SH.coords['y'] = forcing_NH.y
    forcing_NH.coords['latitude'] = np.arcsin(forcing_NH.y)*180.0/np.pi
    forcing_SH.coords['latitude'] = np.arcsin(forcing_SH.y)*180.0/np.pi
    return forcing_NH,forcing_SH
