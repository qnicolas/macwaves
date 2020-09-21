import numpy as np
import scipy.sparse as sps
import scipy.linalg as spl
import matplotlib.pyplot as plt

### Next steps
# try to implement with sparse matrices

##### MODEL PARAMETERS #####

m = 3                              # Angular order
H = 140e3                          # Stratified layer thickness (m)
B = 0.6 * 1.0e-3                   # Radial magnetic field (Tesla)
Np = 0.5                           # dimensionless stratification Np = N/Omega
T = -20.                           # Wave period initial estimate (years)

sectoyear = 365.25 * 24 * 60 * 60
omega0 = 2*np.pi/(T*sectoyear)     # Wave angular frequency initial estimate (rad.s-1)

mu = 4 * np.pi * 1.0e-7            # permeability
sigma = 1.0e6                      # conductivity (S/m)
eta = 1.0 / (mu * sigma);          # diffusivity (m^2/s)
rho = 1.0e4                        # density  (kg/m^3)
R = 3.48e6                         # radius of core (m)
Omega = 0.7292e-4;                 # Earth's rotation rate (s^-1)
N = Np*Omega                       # Layer Brunt-Vaisala frequency (s^-1)
k = np.pi/H                        # vertical wavenumber

Ngrid = 300                        # number of grid points in latitude (equator to pole)

##### DERIVED PARAMETERS #####
# evaluate diffuion factor using estimate of omega
chi = 1  +  eta * k**2 * 1j / omega0
M = B**2 * k**4 * R**2 / (rho * mu * N**2 * chi)
Cp = 2*Omega * k**2 * R**2 / N**2
C0 = Cp*omega0


######## MAIN CODE ########
def secondDerivative(y):
    n = len(y)
    dy = y[1]-y[0]
    return (1/dy**2)*sps.diags([1, -2, 1], [-1, 0, 1], shape=(n,n)).toarray()

def sety(ngrid):
    return np.linspace(-1.,1.,ngrid+2)[1:-1] #exclude the -1 and 1 bounds to avoid divisions by 0

def setA(ngrid,m,M,Cp):
    """ Sets matrix A of the eigenvalue problem (A + i*omega*In)x = F"""
    y  = sety(ngrid)
    D2 = secondDerivative(y)
    L1 = np.diag(M*(1-y**2)/(Cp**2*y**2)) * D2 - np.diag(M*(m**2-1)/(Cp**2 * y**2 * (1-y**2)))

    L2 = np.diag(1j*m/(Cp * y**2))

    return np.block([[np.zeros((ngrid,ngrid)),np.eye(ngrid)],[L1,L2]])

def modes_y(A):
    """Solves the linear eigenvalue problem (A + i*omega*In)x = 0 and returns the corresponding b_theta(y)"""
    w,v = spl.eig(A)
    return 1j*w,v


w,v = modes_y(setA(Ngrid,m,M,Cp))
y = sety(Ngrid)

i = np.argmin((np.real(w) - omega0)**2)

print(w[i])

va = v[i]
_,ax=plt.subplots(1,1,figsize=(20,10))
by = va[:Ngrid]/(1-y**2)
amp = np.max(np.abs(np.real(by)));
ax.plot(y,np.real(by)/amp)
ax.set_xlabel("cos Latitude")
ax.set_ylabel("Wave Amplitude")
plt.show()

print(w)
print(C0)

#D2 = secondDerivative(y)
#L1 = M*(1-y**2)/(Cp**2*y**2) * D2 - M*(m**2-1)/(Cp**2 * y**2 * (1-y**2)) * np.eye(Ngrid)
#print(L1)
