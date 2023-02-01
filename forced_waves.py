from time_tools import sectoyear,nonzero_frequencies
from macmodes import *

    
#############################################################################################
############################### SETTING UP RHS FORCING TERMS ################################
#############################################################################################

def set_rhs_forcing(param,M,chi,freq,forcings,D1D2=None):  ###### DOCSTRING TO BE REWRITTEN ######
    """Computes the right hand side of the forced wave problem, which is a vector of length 2*ngrid
    whose first n components are zero, and next n are functions of the forcing components Fr, Ftheta,
    Flambda, FItheta and FIlambda. See equation A.17 of Nicolas & Buffett (2023).
    
    Parameters
    ----------
    param    : Param object
        Contains model parameters (e.g. spatial scales, rotation rate, stratification strength, etc.).
    M        : complex
        Nondimensional term scaling the magntude of the Lorentz force relative to buoyancy
    chi      : complex
        Nondimensional damping term
    freq     : float
        Frequency (in s^-1) of the wave to be forced.
    forcings : xarray.Dataset
        Forcing object containing different forcing components: Fr, Ftheta, Flambda, FItheta, FIlambda. 
        These components each have dimensions of frequency, radial order and y (meridional coordinate).
    D1D2     : None or tuple of numpy.ndarray
        If a tuple, contains matrices of first and second order differentiation on the meridional grid 
        on which forcings (and waves) are defined. If None, these matrices are computed. Used to avoid
        repeating calculations.
        
    Returns
    -------
    F        : numpy.ndarray
        right hand side of the discrete forced wave problem, of size 2*ngrid
    """
    y = np.array(forcings.y)
    ngrid=len(y)
    
    if D1D2 is None:
        D1 = make_D_fornberg(y,1)
        D2 = make_D_fornberg(y,2)
    else:
        D1,D2=D1D2

    # Compute nondimensional parameters
    C,_,_ = CMchi(param,1/float(freq),float(forcings.radial_order))        # convert to float to speed things up
    Af_h,Af_r = Afactors(param,1/float(freq),float(forcings.radial_order)) # convert to float to speed things up

    # Compute each component of the sum in A.17
    forcingr =  -  (C*param.m/M + 2)*y*Af_r*np.array(forcings.Fr)/(1-y**2)**(0.5) - (1-y**2)**(0.5)*np.dot(D1,Af_r*np.array(forcings.Fr))
    forcingtheta = param.m**2/(M*(1-y**2)) * Af_h*np.array(forcings.Ftheta)
    forcinglambda = -1j*( (C/M + 2/param.m)*y*Af_h*np.array(forcings.Flambda)
                          + param.m/(M*(1-y**2)**(0.5)) * np.dot(D1,Af_h*np.array(forcings.Flambda)*(1-y**2)**(0.5)) 
                         )
    
    FItheta_pp = np.array(forcings.FItheta)*(1-y**2)
    FIlambda_p = np.array(forcings.FIlambda)/(1-y**2)**0.5
    
    forcingItheta = 1/chi*(np.dot(D2,FItheta_pp) + (C**2*y**2+param.m*C)/(M*(1-y**2))*FItheta_pp + (1-y**2)**(-2)*FItheta_pp)
    forcingIlambda =   1/chi*(-1j*param.m*(1-y**2)**(0.5)*np.dot(D1,FIlambda_p) - 1j*C*y*(1-y**2)**(0.5)*FIlambda_p + 2j*param.m*y*(1-y**2)**(-0.5)*FIlambda_p)
    
    # Sum, transform, concatenate with zeroes (see manuscript, right after equation 25)
    rhs = np.concatenate([np.zeros(ngrid),-(forcingr+forcingtheta+forcinglambda+forcingItheta+forcingIlambda)*M*(1-y**2)/y**2])  
    return rhs

#############################################################################################
######################### SOLVING FOR ONE FORCED WAVE COMPONENT #############################
#############################################################################################
    
def solve_forced_problem_nmodes_fast(param,M,chi,freq,j,Ci,pi,xi,forcings,nmodes,D1D2=None):
    """Given model parameters and forcings, compute the meridional structure of one MAC wave 
    mode (See equation 29 of Nicolas & Buffett, 2023)
    
    Parameters
    ----------
    param    : Param object
        Contains model parameters (e.g. spatial scales, rotation rate, stratification strength, etc.).
    M        : complex
        Nondimensional term scaling the magntude of the Lorentz force relative to buoyancy
    chi      : complex
        Nondimensional damping term
    freq     : float
        Frequency (in s^-1) of the wave to be forced.
    j        : int
        Radial order of the wave to be forced.
    Ci   : numpy.ndarray
        Eigenvalues, sorted by increasing period.
    pi       : numpy.ndarray
        Matrix of size (2*ngrid)*(2*ngrid) containing all the left eigenmodes, sorted by increasing 
        period. 
    xi       : numpy.ndarray
        Matrix of size (2*ngrid)*(2*ngrid) containing all the eigenmodes, sorted by increasing period.
        xi and pi are normalized such that p_j^H * xi = delta_ij.
    forcings : xarray.Dataset
        Forcing object containing different forcing components: Fr, Ftheta, Flambda, FItheta, FIlambda. 
        These components each have dimensions of frequency, radial order and y (meridional coordinate).
    nmodes   : int
        Number of eigenmodes to include, where modes are sorted by increasing period (starting with the 
        most negative period, i.e. the gravest high-latitude mode).
    D1D2     : None or tuple of numpy.ndarray
        If a tuple, contains matrices of first and second order differentiation on the meridional grid 
        on which forcings (and waves) are defined. If None, these matrices are computed. Used to avoid
        repeating calculations.

    Returns
    -------
    b        : numpy.array
        Meridional structure of the forced wave (in terms of a meridional magnetic perturbation).
    """
    y=np.array(forcings.y)
    ngrid=len(y)
    C,_,_ = CMchi(param,1/freq,j)
    
    # Compute right-hand side of the discrete forced wave equation
    F = set_rhs_forcing(param,M,chi,freq,forcings.sel(frequency=freq, radial_order=j),D1D2)
    
    # Compute the sum in equation (29) of Nicolas & Buffett (2023)
    b_doubleprime = np.sum(np.dot(pi[:,:nmodes].conj().T,F)/(Ci[:nmodes]-C) * xi[:,:nmodes],axis=1)[:ngrid]
    return b_doubleprime/(1-y**2) #Transform back to a physical magnetic perturbation


#############################################################################################
############################# SOLVING FOR ALL FORCED WAVES ##################################
#############################################################################################

def forced_MAC_wave_spectrum(param,forcings_NH,forcings_SH,jmax=1,nmodes='all',freqindex=-1):
    """Given model parameters and forcings, compute the spatio-temporal structure of forced MAC waves 
    (See equation 18 of Nicolas & Buffett, 2023)
    
    Parameters
    ----------
    param       : Param object
        Contains model parameters (e.g. spatial scales, rotation rate, stratification strength, etc.).
    forcings_NH : xarray.Dataset
        Forcing object containing different forcing components: Fr, Ftheta, Flambda, FItheta, FIlambda. 
        These components each have dimensions of frequency, radial order and y (meridional coordinate),
        and represent the Northern Hemisphere part of each forcing.
    forcings_SH : xarray.Dataset
        Same as forcings_NH, except it contains the Southern Hemisphere part of each forcing.
    jmax        : int
        Maximum radial order to include.
    nmodes      : 'all' or int
        If all, include all wave modes. If an integer, limit the analysis to the first nmode eigenmodes,
        sorted by increasing period (starting with the most negative period, i.e. the gravest 
        high-latitude mode).
    freqindex   : int
        If positive, only consider one frequency. In this case the function returns 0 for the temporal 
        spectrum; only the spatial structure is relevant. If negative, include all frequencies. The 
        latter is the default.

    Returns
    -------
    RMS_b        : xarray.DataArray
        RMS meridional structure of the forced waves (in terms of a meridional magnetic perturbation)
    powspec_hl_b : xarray.DataArray
        Temporal power spectrum of the forced waves, averaged over latitudes poleward of 45° (in terms
        of a meridional magnetic perturbation)
    """
    y = forcings_NH.y
    ngrid = len(y)
    js=forcings_NH.radial_order[:jmax]
    ks = np.pi/param.H*js
    period=-20*sectoyear ## period to compute fixed damping term chi
    
    # set up nmodes
    if nmodes=='all':
        nmodes=2*ngrid # include all eigenmodes
    else:
        assert type(nmodes)==int
    
    # Set up frequencies and indices
    if freqindex >= 0:
        frequencies = nonzero_frequencies()[freqindex:freqindex+1]
    else:
        frequencies=nonzero_frequencies()
    omegas = 2*np.pi*frequencies
    
    #Placeholder for forced waves
    by_all_NH= 0.* forcings_NH.FItheta.sel(frequency = frequencies,radial_order=js).rename('btheta')
    by_all_SH= by_all_NH.copy()
    
    #Precompute discrete derivation matrices
    D1D2=(make_D_fornberg(y,1),
          make_D_fornberg(y,2))

    print("Number of frequencies treated (%i total): "%(len(frequencies)),end=' ')
    for j in np.array(js):
        _,M,chi = CMchi(param,period,j)
        A = set_A(ngrid,param.m,M)
        Ci,pi,xi = compute_eigendecomp(A)
        for i,freq in enumerate(frequencies):
            if i%20==0:
                print(i,end=' ')
            by_all_NH[i,j-1,:] = solve_forced_problem_nmodes_fast(param,M,chi,freq,j,Ci,pi,xi,forcings_NH,nmodes,D1D2=D1D2)
            by_all_SH[i,j-1,:] = solve_forced_problem_nmodes_fast(param,M,chi,freq,j,Ci,pi,xi,forcings_SH,nmodes,D1D2=D1D2)
            
    # Compute RMS meridional structure (average NH and SH)
    RMS_b = 0.5*(np.sqrt(0.5*(np.abs(by_all_NH)**2).sum(['frequency','radial_order']))
                +np.sqrt(0.5*(np.abs(by_all_SH)**2).sum(['frequency','radial_order']))
                )

    if freqindex>=0:
        powspec_hl_b=0
    else:
        # divide by delta Omega and multiply by 2pi to have units of a power spectrum
        deltaomega = omegas[1]-omegas[0]
        
        #powspec_b = 0.5*((0.5*(np.abs(by_all_NH)**2).sum('radial_order') * (np.gradient(y,edge_order=2)*y**0)).sum('y')
        #                +(0.5*(np.abs(by_all_SH)**2).sum('radial_order') * (np.gradient(y,edge_order=2)*y**0)).sum('y')
        #                ) / deltaomega * 2*np.pi
        
        # High-latitude power spectrum (average for y>0.5)
        powspec_hl_b = 0.5*((0.5*(np.abs(by_all_NH)**2).sum('radial_order') * (np.gradient(y,edge_order=2) * (y>1/np.sqrt(2)) / (1-1/np.sqrt(2)))).sum('y')
                           +(0.5*(np.abs(by_all_SH)**2).sum('radial_order') * (np.gradient(y,edge_order=2) * (y>1/np.sqrt(2)) / (1-1/np.sqrt(2)))).sum('y')
                           ) / deltaomega * 2*np.pi
        frequency_yr = powspec_hl_b.frequency*sectoyear
        frequency_yr.attrs['unit']='yr^-1'
        powspec_hl_b=powspec_hl_b.assign_coords(frequency_yr=frequency_yr).sortby('frequency')
        
    return RMS_b,powspec_hl_b
    
