import numpy as np
import scipy.sparse as sps
import scipy.linalg as spl
import scipy.sparse.linalg as spsl
from finitediff import get_weights

class Param:
    def __init__(self, m, H, N, Br, L = 2260e3, Omega = 2*np.pi/86400, eta = 0.8, rho0 = 1e4, mu0 = 4e-7*np.pi):
        """A structure to hold all parameters of the MAC wave model
        
        init Parameters
        ----------
        m : int
            Angular order of the waves
        H : float
            Stratified layer thickness (in m)
        N : float
            Buoyancy frequency in the stratified layer (in s^-1)
        Br : float
            RMS radial component of the magnetic field in Earth's core at high latitudes (in T)
        L : float
            Earth's outer core thickness (in m)
        Omega : float
            Earth's rotation rate (in s^-1)
        eta : float
            Magnetic diffusivity in Earth's outer core (in m^2/s)
        rho0 : float
            Density of the outer core (in kg m^-3)
        mu0 : float
            Permeability of free space (in T m A^-1)
        """ 
        self.m     = m     
        self.H     = H     
        self.N     = N     
        self.Br    = Br    
        self.L     = L     
        self.R     = self.L/0.65  
        self.Omega = Omega 
        self.eta   = eta   
        self.rho0  = rho0  
        self.mu0   = mu0   

def set_y(ngrid,two_sides=False):
    """Sets a Chebyshev grid in the meridional coordinate y (=cos(latitude)).
    
    Parameters
    ----------
    ngrid     : int
        Number of points in one hemisphere.
    two_sides : bool
        Whether to return a hemispheric grid (from 0 to 1) or a full-sphere grid (-1 to 1).
        
    Returns
    -------
    y : numpy.ndarray
        The meridional grid.
    """
    if two_sides:
        return np.cos(np.linspace(0.,np.pi,2*ngrid+2)[1:-1][::-1])
    else:
        return np.cos(np.linspace(0.,np.pi,2*ngrid+2)[1:-1][::-1][ngrid:]) #exclude the right bound to avoid divisions by 0

def make_D_fornberg(y,order,npoints=7):
    """
    Computes a differentiation matrix for the mth derivative on a nonuniform 
    grid, using a npoints-point stencil (uses Fornberg's algorithm)
    
    Parameters
    ----------
    y       : numpy.ndarray
        Grid.
    order   : int
        Order of differentiation.
    npoints : int
        Number of points for the finite difference stencil
        
    Returns
    -------
    D : numpy.ndarray, 2D
        Differentiation matrix.
    """
    N=len(y)
    assert N>=npoints
    D=np.zeros((N,N))
    for i in range(npoints//2):
        D[ i,:npoints] = get_weights(y[:npoints],y[i],-1,order)[:,order]
        D[-i-1,-npoints:] = get_weights(y[-npoints:],y[-i-1],-1,order)[:,order] 
    for i in range(npoints//2,N-npoints//2):
        D[i,i-npoints//2:i+npoints//2+1] = get_weights(y[i-npoints//2:i+npoints//2+1],y[i],-1,order)[:,order]   
    return D

def set_A(ngrid,m,M,BC="Dirichlet"):
    """ Sets the matrix A of the MAC wave eigenvalue problem (A - C*In)x = 0
    
    Parameters
    ----------
    ngrid   : int
        Number of points in one hemisphere.
    m       : int
        Angular order of the waves.
    M       : complex
        Nondimensional number scaling the ratio of Lorentz force to buoyancy force (See Nicolas & Buffett 2023).
 (matrix A contains a second order derivative)
    BC      : str
        Either "Dirichlet" or "Neumann", which boundary condition to use at the equator.
        
    Returns
    -------
    A : numpy.ndarray, 2D
        matrix of the MAC wave eigenvalue problem on a hemispherical grid.
    """
    y  = set_y(ngrid)
    D2 = make_D_fornberg(y,2) #Matrix of second differentiation
    
    L1 = -np.dot(np.diag(M*(1-y**2)/(y**2)), D2) + np.diag(M*(m**2-1)/(y**2 * (1-y**2)))
    L2 = -np.diag(m/(y**2))
    
    A=np.block([[np.zeros((ngrid,ngrid)),np.eye(ngrid)],[L1,L2]])
    
    #Enforce boundary conditions
    A[len(y)] = np.zeros(len(A))
    A[-1] = np.zeros(len(A))
    scale=np.abs(A[len(y)+1,0])
    if BC == "Dirichlet":
        A[len(y),0]=   1
        A[-1,len(y)-1]=1
    elif BC == "Neumann": #second-order one-sided finite difference @ equator
        A[len(y),:3]  = np.array([-3,4,-1])
        A[-1,len(y)-1]= 1
    else:
        raise ValueError("BC must be either Dirichlet or Neumann")
    # The following avoids numerical errors at the boundaries, that appear when the boundary condition is specified too weakly
    A[len(y)] *= scale
    A[-1]    *= scale
    return A

def Afactors(param,period,j):
    """Computes prefactor A and A/(R*k) for forcing terms (See equation A.13 of Nicolas & Buffett, 2023)
    
    Parameters
    ----------
    param  : Param object
        Contains model parameters.
    period : float
        Period of the waves of interest. Negative indicates westward-traveling waves.
    j      : int
        Radial order of the waves.

    Returns
    -------
    Af_h : complex
        Forcing constant for horizontal forcings
    Af_r : complex
        Forcing constant for radial forcings
    """
    k = np.pi*j/param.H
    omega = 2*np.pi / period
    chi = 1 +1j*param.eta*k**2/omega
    Af_h = param.R**2*k**3*param.Br/(chi*param.N**2)
    return np.array(Af_h), np.array(Af_h/(param.R*k))  #Convert to numpy.array in case one of the arguments was a xarray.DataArray object
    
def CMchi(param,period,j):
    """Computes coefficients C,M and damping term chi (See equations 19 and A.8 of Nicolas & Buffett, 2023).
    
    Parameters
    ----------
    param  : Param object
        Contains model parameters.
    period : float
        Period of the waves of interest. Negative indicates westward-traveling waves.
    j      : int
        Radial order of the waves.

    Returns
    -------
    C   : complex
        Nondimensional term scaling the magntude of the Coriolis force relative to buoyancy
    M   : complex
        Nondimensional term scaling the magntude of the Lorentz force relative to buoyancy
    chi : complex
        Nondimensional damping term
    """
    k = np.pi*j/param.H
    omega = 2*np.pi / period
    chi = 1 +1j*param.eta*k**2/omega
    C = 2*param.Omega*omega*k**2*param.R**2/(param.N**2)
    M = param.Br**2*k**4*param.R**2/(param.rho0*param.mu0*chi*param.N**2)
    return np.array(C),np.array(M),np.array(chi) #Convert to numpy.array in case one of the arguments was a xarray.DataArray object

def compute_eigendecomp(A):
    """Given a matrix A, compute eigendecomposition, sort results by increasing period 
    (starting with the most negative period) and return eigenvalues C_i, left eigenvectors 
    p_i (normalized such that p_j^H * x_i = delta_ij), and right eigenvectors x_i.
    
    Parameters
    ----------
    A  : numpy.ndarray, 2D
        Square complex matrix to be decomposed

    Returns
    -------
    Ci   : numpy.ndarray
        Sorted eigenvalues
    pi   : numpy.ndarray
        Left eigenvectors, same shape as A
    xi   : numpy.ndarray
        Right eigenvectors, same shape as A
    """
    Ci,pi,xi=spl.eig(A,left=True)
    #Normalize left eigenvectors so that pj^H * xi = delta_ij
    pi = pi/np.diagonal(np.dot(pi.transpose().conj(),xi))[None,:].conj()
    #Sort eigenvectors by increasing period
    order=np.argsort(np.real(1/Ci))
    xi=xi[:,order]
    pi=pi[:,order]
    Ci=Ci[order]    
    return Ci,pi,xi
    


