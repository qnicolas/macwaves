import numpy as np
import scipy.sparse as sps
import scipy.linalg as spl
import scipy.sparse.linalg as spsl
import xarray as xr
from analysisTools import *
from solvingAveragingTools import *




def solve_forced_problem_gev(y,A,B,C,rhs):
    ngrid=len(y)
    #print(np.linalg.cond(A-C*B),np.linalg.cond(A),np.linalg.cond(B))
    A = sps.csc_matrix(A)
    B = sps.csc_matrix(B)
    return spsl.spsolve(A-C*B,rhs)[:ngrid]/(1-y**2)

def solve_forced_problem_Nmodes_gev(y,A,zB,C,rhs,nmodes):
    ngrid=len(y)
    n0=0
    wi,zi,xi=spl.eig(A,B,left=True)
    #Normalize left eigenvectors so that zj^H * xi = delta_ij
    zi = zi/np.diagonal(np.dot(zi.transpose().conj(),np.dot(B,xi)))[None,:].conj()
    #Sort eigenvectors by increasing period (on)
    order=np.argpartition(np.real(1/wi),n0+nmodes)
    xi=xi[:,order]
    zi=zi[:,order]
    wi=wi[order]
    return np.sum(np.dot(zi[:,n0:n0+nmodes].conj().T,rhs)/(wi[n0:n0+nmodes]-C) * xi[:,n0:n0+nmodes],axis=1)[:ngrid]/(1-y**2)

def set_solve_forced_problem_gev(forcing_name,forcingtilda_r,forcingtilda_theta,forcingtilda_phi,m,freqindex,j,fixedparams,npoints=7,limitmodes=False,OPT=1):
    """Given a fourier transformed forcing, problem parameters, and a given frequency index and radial wavenumber, compute the forced wave response.
        - forcing_name       : str, either "buoyancy", "lorentz"
        - forcingtilda_r     : numpy.ndarray, temperature OR r-component of the Lorentz force, transformed in time and radius. Unused if forcingname=="induction". Dimensions are frequency, radial wavenumber, and y (=cos(latitude))
        - forcingtilda_theta : numpy.ndarray, unused if forcing_name is "buoyancy" OR theta-component of the Lorentz force OR theta-component of the magnetic field, transformed in time and radius. Dimensions are frequency, radial wavenumber, and y (=cos(latitude))
        - forcingtilda_phi   : numpy.ndarray, unused if forcing_name is "buoyancy" OR phi-component of the Lorentz force  OR phi-component of the magnetic field, transformed in time and radius. Dimensions are frequency, radial wavenumber, and y (=cos(latitude))
        - m                  : int, spherical harmonic order 
        - freqindex          : int, index of the frequency to pick.
        - j                  : int, radial wavenumber
        - fixedparams        : float iterable, contains in order E,Pm,Rastar,Hprime,Nprime,Brprime
        - npoints            : number of points for the finite difference stencil (forcing terms are differentiated to compute the rhs)
    returns :
        - numpy.ndarray of length ngrid, solution of the forced wave problem (magnetic perturbation \tilde b_theta)
    """
    # Compute C and M
    ngrid = forcingtilda_r.shape[2]  
    period = freqindex_to_period(freqindex)
    C,M,chi = CMchi(fixedparams,period,j)
    
    # Set up unforced problem
    y = set_y(ngrid)
    A,B = set_AB(ngrid,m,M,npoints)
    
    # Compute forcings & setup RHS
    if forcing_name == "induction":
        FItheta = forcingtilda_theta[freqindex,j-1,:]
        FIphi = forcingtilda_phi[freqindex,j-1,:]
        zero=0*FIphi
        rhs = set_rhs_forcing_gev(y,C,m,M,chi,zero,zero,zero,FItheta,FIphi,npoints)
    else:
        if forcing_name == "buoyancy":
            Fr,Ftheta,Fphi = set_buoyancy_forcing(forcingtilda_r,freqindex,j,fixedparams)
        elif forcing_name == "lorentz":
            Fr,Ftheta,Fphi = set_lorentz_forcing(forcingtilda_r,forcingtilda_theta,forcingtilda_phi,freqindex,j,fixedparams)
        zero=0*Fr
        rhs = set_rhs_forcing_gev(y,C,m,M,chi,Fr,Ftheta,Fphi,zero,zero,npoints)
    
    # Solve forced problem
    if limitmodes:
        return solve_forced_problem_Nmodes_gev(y,A,B,C,rhs,limitmodes)
    else:
        return solve_forced_problem_gev(y,A,B,C,rhs)




def set_AB(ngrid,m,M,npoints=7):
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
    
    L1 = -np.dot(np.diag(M*(1-y**2)), D2) + np.diag(M*(m**2-1)/(1-y**2))
    L2 = -m*np.eye(ngrid)
    
    A = np.block([[np.zeros((ngrid,ngrid)),np.eye(ngrid)],[L1,L2]])
    B = np.block([[np.eye(ngrid),np.zeros((ngrid,ngrid))],[np.zeros((ngrid,ngrid)),np.diag(y**2)]])
    
    #Enforce Dirichlet boundary conditions
    A[len(y)] = np.zeros(len(A))
    A[-1] = np.zeros(len(A))
    scale=np.abs(A[len(y)+1,0])
    A[len(y),0]=   1
    A[-1,len(y)-1]=1
    # The following avoids numerical errors at the boundaries, that appear when the boundary condition is specified too weakly
    A[len(y)] *= scale**2
    A[-1]    *= scale**2
    
    
    return A,B

def set_rhs_forcing_gev(y,C,m,M,chi,Fr,Ftheta,Fphi,FItheta,FIphi,npoints=7):
    ngrid=len(y)
    
    D1 = make_D_fornberg(y,1,npoints=npoints)
    D2 = make_D_fornberg(y,2,npoints=npoints)
    
    forcingr =  -  (C*m/M + 2)*y*Fr/(1-y**2)**(0.5) - (1-y**2)**(0.5)*np.dot(D1,Fr)
    forcingtheta = m**2/(M*(1-y**2)) * Ftheta
    forcingphi = -1j* ( (C/M + 2/m)*y*Fphi
                          + m/(M*(1-y**2)**(0.5)) * np.dot(D1,Fphi*(1-y**2)**(0.5)) 
                         )
    
    FItheta_pp = FItheta*(1-y**2)
    FIphi_p = FIphi/(1-y**2)**0.5
    
    forcingItheta = 1/chi*(np.dot(D2,FItheta_pp) + (C**2*y**2+m*C)/(M*(1-y**2))*FItheta_pp + (1-y**2)**(-2)*FItheta_pp)
    forcingIphi =   1/chi*(-1j*m*(1-y**2)**(0.5)*np.dot(D1,FIphi_p) - 1j*C*y*(1-y**2)**(0.5)*FIphi_p + 2j*m*y*(1-y**2)**(-0.5)*FIphi_p)
    
    rhs = np.concatenate([np.zeros(ngrid),-(forcingr+forcingtheta+forcingphi+forcingItheta+forcingIphi)*M*(1-y**2)])  
    return rhs