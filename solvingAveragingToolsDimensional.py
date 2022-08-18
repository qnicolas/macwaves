import numpy as np
import scipy.sparse as sps
import scipy.linalg as spl
import scipy.sparse.linalg as spsl
import xarray as xr
from analysisTools import *


import time


#############################################################################################
############################## SETTING UP GRID AND WAVE MATRIX ##############################
#############################################################################################

def set_y(ngrid,two_sides=False):
    """Sets a y (=cos(latitude)) grid from 0 to 1, nonlinearly spaced, with ngrid points"""
    if two_sides:
        return np.cos(np.linspace(0.,np.pi,2*ngrid+2)[1:-1][::-1])
    else:
        return np.cos(np.linspace(0.,np.pi,2*ngrid+2)[1:-1][::-1][ngrid:]) #exclude the right bound to avoid divisions by 0

def set_A(ngrid,m,M,npoints=7,option=1):
    """ Sets the matrix A of the wave eigenvalue problem (A - C*In)x = 0
    args :
        - ngrid   : int, number of points in y
        - m       : int, spherical harmonic order 
        - M       : complex, nondimensional number scaling Magnetic force vs buoyancy force in the MAC balance
        - npoints : number of points for the finite difference stencil (A contains a second order derivative)
    returns :
        - numpy.ndarray of the same shape as x except along the axis, where its length is incremented by 1
    """
    y  = set_y(ngrid)
    D2 = make_D_fornberg(y,2,npoints=npoints) #Matrix of second differentiation
    
    L1 = -np.dot(np.diag(M*(1-y**2)/(y**2)), D2) + np.diag(M*(m**2-1)/(y**2 * (1-y**2)))
    L2 = -np.diag(m/(y**2))
    
    A=np.block([[np.zeros((ngrid,ngrid)),np.eye(ngrid)],[L1,L2]])
    #Enforce Dirichlet boundary conditions
    A[len(y)] = np.zeros(len(A))
    A[-1] = np.zeros(len(A))
    scale=np.abs(A[len(y)+1,0])
    if option == 1: #Dirichlet
        A[len(y),0]=   1
        A[-1,len(y)-1]=1
    elif option == 2: #Neumann; second-order one-sided finite difference @ equator
        A[len(y),:3]  = np.array([-3,4,-1])
        A[-1,len(y)-1]= 1
    # The following avoids numerical errors at the boundaries, that appear when the boundary condition is specified too weakly
    A[len(y)] *= scale
    A[-1]    *= scale
    return A

def Afactors(param,period,j):
    """Computes prefactor A and A/(R*k) for forcing terms"""
    k = np.pi*j/param.H
    omega = 2*np.pi / period
    chi = 1 +1j*param.eta*k**2/omega
    A = param.R**2*k**3*param.Br/(chi*param.N**2)
    return np.array(A), np.array(A/(param.R*k))
    
def CMchi(param,period,j):
    """Computes C,M and damping term chi"""
    k = np.pi*j/param.H
    omega = 2*np.pi / period
    chi = 1 +1j*param.eta*k**2/omega
    C = 2*param.Omega*omega*k**2*param.R**2/(param.N**2)
    M = param.Br**2*k**4*param.R**2/(param.rho0*param.mu0*chi*param.N**2)
    return np.array(C),np.array(M),np.array(chi)

def compute_eigendecomp(A):
    """Computes eigenvalues, right and left eigenvectors and normalizes"""
    wi,zi,xi=spl.eig(A,left=True)
    #Normalize left eigenvectors so that zj^H * xi = delta_ij
    zi = zi/np.diagonal(np.dot(zi.transpose().conj(),xi))[None,:].conj()
    #Sort eigenvectors by increasing period (on)
    order=np.argsort(np.real(1/wi))
    xi=xi[:,order]
    zi=zi[:,order]
    wi=wi[order]    
    return wi,zi,xi
    

    
    
    
    
    
#############################################################################################
############################### SETTING UP RHS FORCING TERMS ################################
#############################################################################################

def set_rhs_forcing(param,freq,M,chi,forcings,npoints=7,D1D2=None):  ###### DOCSTRING TO BE REWRITTEN ######
    """Computes the right hand side of the forced wave problem, which is a vector of length 2*ngrid
    whose first n components are zero, and next n are functions of the nondimensional forcings Fr, Ftheta and Fphi
    args :
        - y          : numpy.array, grid (length=ngrid)
        - C          : float, nondimensional number scaling Coriolis force vs buoyancy force in the MAC balance. Also acts as the eigenvalue parameter in this eigenvalue problem (it is proportional to the wave frequency) 
        - m          : int, spherical harmonic order 
        - M          : complex, nondimensional number scaling Magnetic force vs buoyancy force in the MAC balance
        - chi        : complex, damping coefficient
        - Fr         : numpy.array, r component of the nondimensional forcing term (length=ngrid)
        - Ftheta     : numpy.array, theta component of the nondimensional forcing term (length=ngrid)
        - Fphi       : numpy.array, phi component of the nondimensional forcing term (length=ngrid)
        - FItheta : numpy.array, theta component of the time integral of the nondimensional induction forcing term (length=ngrid, this is typically the theta-component of the large-scale magnetic field times 1-y^2)
        - FIphi    : numpy.array, phi component of the time integral of the nondimensional induction forcing term (length=ngrid, this is typically the phi-component of the large-scale magnetic field divided by sqrt(1-y^2))
        - npoints : number of points for the finite difference stencil (forcing terms are differentiated to compute the rhs)
    returns :
        - numpy.ndarray of length ngrid, right hand side of the discretized wave equation
    """
    #t0=time.time()
    y = np.array(forcings.y)
    ngrid=len(y)
    
    if D1D2 is None:
        D1 = make_D_fornberg(y,1,npoints=npoints)
        D2 = make_D_fornberg(y,2,npoints=npoints)
    else:
        D1,D2=D1D2

    C,_,_ = CMchi(param,1/float(freq),float(forcings.radial_order))        # convert to float to speed things up
    Af_h,Af_r = Afactors(param,1/float(freq),float(forcings.radial_order)) # convert to float to speed things up

    forcingr =  -  (C*param.m/M + 2)*y*Af_r*np.array(forcings.Fr)/(1-y**2)**(0.5) - (1-y**2)**(0.5)*np.dot(D1,Af_r*np.array(forcings.Fr))
    forcingtheta = param.m**2/(M*(1-y**2)) * Af_h*np.array(forcings.Ftheta)
    forcingphi = -1j*( (C/M + 2/param.m)*y*Af_h*np.array(forcings.Fphi)
                          + param.m/(M*(1-y**2)**(0.5)) * np.dot(D1,Af_h*np.array(forcings.Fphi)*(1-y**2)**(0.5)) 
                         )
    
    FItheta_pp = np.array(forcings.FItheta)*(1-y**2)
    FIphi_p = np.array(forcings.FIphi)/(1-y**2)**0.5
    
    forcingItheta = 1/chi*(np.dot(D2,FItheta_pp) + (C**2*y**2+param.m*C)/(M*(1-y**2))*FItheta_pp + (1-y**2)**(-2)*FItheta_pp)
    forcingIphi =   1/chi*(-1j*param.m*(1-y**2)**(0.5)*np.dot(D1,FIphi_p) - 1j*C*y*(1-y**2)**(0.5)*FIphi_p + 2j*param.m*y*(1-y**2)**(-0.5)*FIphi_p)
    
    #print('comput',time.time()-t0);t0=time.time()
    rhs = np.concatenate([np.zeros(ngrid),-(forcingr+forcingtheta+forcingphi+forcingItheta+forcingIphi)*M*(1-y**2)/y**2])  
    return rhs







# #############################################################################################
# ################ WAVE SOLUTION FOR ONE RADIAL WAVENUMBER AND ONE FREQUENCY ##################
# #############################################################################################
# 
# def solve_forced_problem(y,A,C,rhs):
#     """Given the matrix A, scaled frequency C and right hand side, compute the solution of the forced wave problem.
#        Note the solution is returned as the magnetic perturbation \tilde b_theta, instead of its transformed analog 
#        that appears in the wave equation represented by A, which is \tilde b_theta'' = \tilde b_theta * (1-y**2)
#         - y   : numpy.array, grid (length=ngrid)
#         - A   : numpy.ndarray, matrix of the wave problem as computed by set_A (size (2*ngrid,2*ngrid))
#         - C   : float, nondimensional number scaling Coriolis force vs buoyancy force in the MAC balance. Also acts as the eigenvalue parameter in this eigenvalue problem (it is proportional to the wave frequency) 
#         - rhs : numpy.array, right hand side of the forced problem as computed by set_rhs_forcing (length=ngrid)
#     returns :
#         - numpy.ndarray of length ngrid, solution of the forced wave problem (magnetic perturbation \tilde b_theta)
#     """
#     ngrid=len(y)
#     A = sps.csc_matrix(A)
#     return spsl.spsolve(A-C*sps.eye(2*ngrid),rhs)[:ngrid]/(1-y**2)
# 
# def solve_forced_problem_Nmodes(y,A,C,rhs,nmodes):
#     ngrid=len(y)
#     wi,zi,xi=compute_eigendecomp(A)
#     return np.sum(np.dot(zi[:,:nmodes].conj().T,rhs)/(wi[:nmodes]-C) * xi[:,:nmodes],axis=1)[:ngrid]/(1-y**2)
# 
# #Here we compute, for a given mode $m,j,\omega$ (with $j=1,2,3$ and $\omega=2\pi/(-20 \mathrm{ years})$), the wave solution as a zonal magnetic perturbation $\tilde b_y(y)$.
# def set_solve_forced_problem(forcing_name,forcingtilda_r,forcingtilda_theta,forcingtilda_phi,m,freqindex,j,fixedparams,npoints=7,limitmodes=0,fixed_damping=False):
#     """Given a fourier transformed forcing, problem parameters, and a given frequency index and radial wavenumber, compute the forced wave response.
#         - forcing_name       : str, either "buoyancy", "lorentz" or 'induction'
#         - forcingtilda_r     : numpy.ndarray, temperature OR r-component of the Lorentz force, transformed in time and radius. Unused if forcingname=="induction". Dimensions are frequency, radial wavenumber, and y (=cos(latitude))
#         - forcingtilda_theta : numpy.ndarray, unused if forcing_name is "buoyancy" OR theta-component of the Lorentz force OR theta-component of the magnetic field, transformed in time and radius. Dimensions are frequency, radial wavenumber, and y (=cos(latitude))
#         - forcingtilda_phi   : numpy.ndarray, unused if forcing_name is "buoyancy" OR phi-component of the Lorentz force  OR phi-component of the magnetic field, transformed in time and radius. Dimensions are frequency, radial wavenumber, and y (=cos(latitude))
#         - m                  : int, spherical harmonic order 
#         - freqindex          : int, index of the frequency to pick.
#         - j                  : int, radial wavenumber
#         - fixedparams        : float iterable, contains in order E,Pm,Rastar,Hprime,Nprime,Brprime
#         - npoints            : number of points for the finite difference stencil (forcing terms are differentiated to compute the rhs)
#         - limitmodes         : int, number of eigenmodes on which to project the solution. If 0, pick all eigenmodes.
#     returns :
#         - numpy.ndarray of length ngrid, solution of the forced wave problem (magnetic perturbation \tilde b_theta)
#     """
#     # Compute C and M
#     ngrid = forcingtilda_r.shape[2]  
#     period = freqindex_to_period(freqindex)
#     C,M,chi = CMchi(fixedparams,period,j)
#     
#     ###QN: temporary
#     if fixed_damping:
#         _,M,chi = CMchi(fixedparams,-20,j)
# 
#     # Set up unforced problem
#     y = set_y(ngrid)
#     A = set_A(ngrid,m,M,npoints)
#     
#     # Compute forcings & setup RHS
#     if forcing_name == "induction":
#         FItheta = forcingtilda_theta[freqindex,j-1,:]
#         FIphi = forcingtilda_phi[freqindex,j-1,:]
#         zero=0*FIphi
#         rhs = set_rhs_forcing(y,C,m,M,chi,zero,zero,zero,FItheta,FIphi,npoints)
#     else:
#         if forcing_name == "buoyancy":
#             Fr,Ftheta,Fphi = set_buoyancy_forcing(forcingtilda_r,freqindex,j,fixedparams)
#         elif forcing_name == "lorentz":
#             Fr,Ftheta,Fphi = set_lorentz_forcing(forcingtilda_r,forcingtilda_theta,forcingtilda_phi,freqindex,j,fixedparams)
#         zero=0*Fr
#         rhs = set_rhs_forcing(y,C,m,M,chi,Fr,Ftheta,Fphi,zero,zero,npoints)
#     # Solve forced problem
#     if limitmodes:
#         return solve_forced_problem_Nmodes(y,A,C,rhs,limitmodes)
#     else:
#         return solve_forced_problem(y,A,C,rhs)
# 
#     
#     
#     
#     
#     
#     
#     
#     
#     
#     
#     
# ##################################################################################################################
# ############ WAVE SOLUTION FOR ONE RADIAL WAVENUMBER AND ALL FREQUENCIES, PROJECTED ON ONE EIGENMODE #############
# ##################################################################################################################
# 
# def spectrum_forced_problem(forcing_name,forcingtilda_r,forcingtilda_theta,forcingtilda_phi,m,eigenmode_number,j,fixedparams,npoints=7):
#     """Compute the power spectrum of forced waves projected on a given eigenmode of A
#         - forcing_name       : str, either "buoyancy", "lorentz" or 'induction'
#         - forcingtilda_r     : numpy.ndarray, temperature OR r-component of the Lorentz force, transformed in time and radius. Unused if forcingname=="induction". Dimensions are frequency, radial wavenumber, and y (=cos(latitude))
#         - forcingtilda_theta : numpy.ndarray, unused if forcing_name is "buoyancy" OR theta-component of the Lorentz force OR theta-component of the magnetic field, transformed in time and radius. Dimensions are frequency, radial wavenumber, and y (=cos(latitude))
#         - forcingtilda_phi   : numpy.ndarray, unused if forcing_name is "buoyancy" OR phi-component of the Lorentz force  OR phi-component of the magnetic field, transformed in time and radius. Dimensions are frequency, radial wavenumber, and y (=cos(latitude))
#         - m                  : int, spherical harmonic order 
#         - eigenmode_number   : int, index of the eigenmode on which to project the waves (in the order of increasing eigenperiods)
#         - j                  : int, radial wavenumber
#         - fixedparams        : float iterable, contains in order E,Pm,Rastar,Hprime,Nprime,Brprime
#         - npoints            : number of points for the finite difference stencil (forcing terms are differentiated to compute the rhs)
#     returns :
#         - sorted frequencies, from most negative to most positive. numpy.ndarray of length nfreqs (=len(forcingtilda_r))
#         - power spectrum, numpy.ndarray of length nfreqs
#         - period of the eigenmode defined by eigenmode_number in years
#     """
#     # Compute C and M
#     ngrid = forcingtilda_r.shape[2]  
#     period = -20 ## Assumes fixed damping !!
#     C,M,chi = CMchi(fixedparams,period,j)
#     
#     # Compute matrix and eigendecomposition; we make the approximation that eigenmodes don't change much if chi is changed
#     y = set_y(ngrid)
#     A = set_A(ngrid,m,M)
#     wi,zi,xi=compute_eigendecomp(A)
#     
#     power_spectrum=[]
#     frequencies=nonzero_frequencies()
#     for freqindex in range(len(frequencies)):
#         # Compute forcings
#         C1,M1,chi1 = CMchi(fixedparams,freqindex_to_period(freqindex),j)
# 
#         if forcing_name == "induction":
#             FItheta = forcingtilda_theta[freqindex,j-1,:]
#             FIphi = forcingtilda_phi[freqindex,j-1,:]
#             zero=0*FIphi
#             rhs = set_rhs_forcing(y,C1,m,M1,chi1,zero,zero,zero,FItheta,FIphi,npoints)
#         else:
#             if forcing_name == "buoyancy":
#                 Fr,Ftheta,Fphi = set_buoyancy_forcing(forcingtilda_r,freqindex,j,fixedparams)
#             elif forcing_name == "lorentz":
#                 Fr,Ftheta,Fphi = set_lorentz_forcing(forcingtilda_r,forcingtilda_theta,forcingtilda_phi,freqindex,j,fixedparams)
#             zero=0*Fr
#             rhs = set_rhs_forcing(y,C1,m,M1,chi1,Fr,Ftheta,Fphi,zero,zero,npoints)
# 
#         c_forcing = np.dot(zi[:,eigenmode_number].conj(),rhs)
#         power=np.abs((c_forcing/(wi[eigenmode_number]-C1)))**2
#         power_spectrum.append(power)
#         
#     power_spectrum=np.array(power_spectrum)[np.argsort(frequencies)]
# 
#     C0 = 2/0.65**2*kprime**2/Nprime**2
#     eigenmode_period = np.real(2*np.pi*C0/(wi[eigenmode_number]*sectoyear*Omega))
#     
#     return np.sort(frequencies),power_spectrum,eigenmode_period
