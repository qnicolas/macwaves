import numpy as np
import xarray as xr

sectoyear=365.25 * 24 * 60 * 60

def ntsteps_to_seconds(n):
    """
    Convert a number of Calypso time steps to dimensional seconds
    """
    t_step = 1e-2 # tstep diffusionless = tstep_calypso / E_calypso = 1e-7/1e-5 = 1e-2
    Omega_adjusted = 4.405286343612335e-08#4.4e-8
    return n * t_step / Omega_adjusted

def seconds_to_ntsteps(s):
    """
    Convert dimensional seconds to a number of Calypso time steps
    """
    t_step = 1e-2 # tstep diffusionless = tstep_calypso / E_calypso = 1e-7/1e-5 = 1e-2
    Omega_adjusted = 4.405286343612335e-08#4.4e-8
    return s/(t_step / Omega_adjusted)

def nonzero_frequencies():
    """
    Get the nonzero frequencies (in s^-1) of a Fourier transformed array (where 
    the original array was on a time grid of 800 times with dt=25 Calypso time 
    steps)
    
    Returns
    -------
    freq : xarray.DataArray
        Array of nonzero frequencies.
    """
    freqs = np.fft.fftfreq(800,ntsteps_to_seconds(25))[1:]
    return xr.DataArray(freqs,coords={'frequency':freqs},dims = ['frequency'],attrs={'unit':'s^-1'})

def freqindex_to_period(idx):
    """
    Goes from the index of a Fourier transformed array (where the original array 
    was on a time grid of 800 times with dt=25 Calypso time steps) to a 
    dimensional period in seconds
    
    Parameters
    ----------
    idx : int
        Frequency index.
    
    Returns
    -------
    period: float
        The corresponding period in seconds.
    """
    return 1/np.array(nonzero_frequencies())[idx]

def period_to_freqindex(period):
    """
    Goes from a dimensional period to the closest corresponding frequency index 
    (where the original array was on a time grid of 800 times with dt=25 Calypso 
    time steps)
    
    Parameters
    ----------
    period: float
        Period in seconds.
    
    Returns
    -------
    period: float
        The index corresponding to the closest frequency.
    """
    return np.argmin((1/np.array(nonzero_frequencies())-period)**2)