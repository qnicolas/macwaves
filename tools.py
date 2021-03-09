import numpy as np
import scipy.sparse as sps
import scipy.linalg as spl
import scipy.sparse.linalg as spsl
import pandas as pd
import xarray as xr
import time

from pyshtools.legendre import PlmSchmidt_d1


###########################################
######     Files preprocessing     ########
###########################################
def dat_to_xarray(file):
    """
    args :
        - file : str, path to a calypso output file
    returns :
        - ds : xarray.dataset, where dimensions have been automatically extracted
       """
    f=open(file,"r")
    header=0
    while f.readline()[:6]!="t_step":
        header+=1
    df = pd.read_table(file,header = header,delimiter = '\s+')
    all_coords=['t_step','radius_ID','degree','order','radial_id','diff_deg_order']
    coords = [cname for cname in df.columns if cname in all_coords]
    ds=df.set_index(coords).to_xarray()
    if 't_step' in coords:
        ds = ds.assign_coords({'time':ds.time.isel({c:0 for c in coords if c != 't_step'}, drop=True)})
    if 'radius_ID' in coords:
        ds = ds.assign_coords({'radius':ds.radius.isel({c:0 for c in coords if c != 'radius_ID'}, drop=True)})
    return ds


######################################################################################
######  Torpol / spherical harmonics to spherical coordinates (visualization)  #######
######################################################################################

PlmSchmidt_d1_temp=np.vectorize(PlmSchmidt_d1,otypes=[np.ndarray],excluded=[0])
def PlmSchmidt_d1_vec(l,m,y) :
    allplm = PlmSchmidt_d1_temp(l,y)
    return np.stack( allplm, axis=0 )[:,:,(l*(l+1))//2+m]

def cap(theta,lbound,ubound,eps):
    return theta * (theta<ubound-eps) * (theta>lbound+eps) + (ubound-eps) * (theta>=ubound-eps) + (lbound+eps) * (theta<=lbound+eps)

### Fixed r, fixed l, all m ###
###############################
def torpol_to_thetaphi_sph_fixedrl(theta,phi,T,dSdr,r,l):
    """ - theta, phi have shapes (nPhi,nTheta) (grid)
        - T,dSdR have shapes (2l+1,) (toroidal / d(poloidal)/dr components of degree l and all orders)
        - l is int (degree)
        - r is float (radius)
        """
    T,dSdr = np.array(T),np.array(dSdr)
    
    theta_vector = cap(theta[0],0.,2*np.pi,1e-6)
    sintheta = np.sin(theta_vector)[:,None]
    y=np.cos(theta_vector)
    y = y - 1e-6 * (y==1.)
    
    allplm = np.stack( PlmSchmidt_d1_temp(l,y), axis=0 )[:,:,(l*(l+1))//2:(l*(l+1))//2+l+1]  
    alldplmdy = allplm[:,1,:]
    alldplmdtheta = - sintheta * alldplmdy  # shape = (nTheta,l+1)
    allplm = allplm[:,0,:]                  # shape = (nTheta,l+1)
    
    ms = np.arange(-l,l+1,1)
    mge0=ms[l:]
    ml0=ms[:l]

    Ftheta_sin_mge0 = -1/(r*sintheta) * mge0 * T[l:] * allplm     # shape = (nTheta,l+1)
    Ftheta_cos_mge0 = 1/r * dSdr[l:] * alldplmdtheta
    Ftheta_cos_ml0  = 1/(r*sintheta) * ml0 * T[:l] * allplm[:,1:][:,::-1]  # shape = (nTheta,l)
    Ftheta_sin_ml0  = 1/r * dSdr[:l] * alldplmdtheta[:,1:][:,::-1]
    
    Fphi_sin_mge0 = -1/(r*sintheta) * mge0 * dSdr[l:] * allplm
    Fphi_cos_mge0 = -1/r * T[l:] * alldplmdtheta
    Fphi_cos_ml0  = 1/(r*sintheta) * ml0 * dSdr[:l] * allplm[:,1:][:,::-1]
    Fphi_sin_ml0  = -1/r * T[:l] * alldplmdtheta[:,1:][:,::-1]    
         
    Ftheta_cos=np.hstack([Ftheta_cos_ml0,Ftheta_cos_mge0])  # shape = (nTheta,2l+1)
    Ftheta_sin=np.hstack([Ftheta_sin_ml0,Ftheta_sin_mge0])
    
    Fphi_cos=np.hstack([Fphi_cos_ml0,Fphi_cos_mge0])
    Fphi_sin=np.hstack([Fphi_sin_ml0,Fphi_sin_mge0])


    Ftheta = Ftheta_cos[None,:,:]*np.cos(ms[None,None,:]*phi[:,:,None]) + Ftheta_sin[None,:,:]*np.sin(ms[None,None,:]*phi[:,:,None])  # shape = (nPhi,nTheta,2l+1)
    Fphi = Fphi_cos[None,:,:]*np.cos(ms[None,None,:]*phi[:,:,None]) + Fphi_sin[None,:,:]*np.sin(ms[None,None,:]*phi[:,:,None])

    return Ftheta.sum(axis=2),Fphi.sum(axis=2)

def pol_to_r_sph_fixedrl(theta,phi,S,r,l):
    """ - theta, phi have shapes (nPhi,nTheta) (grid)
    - S has shapes (2l+1,) (poloidal component of degree l and all orders)
    - l is int (degree)
    - r is float (radius)
    """
    S = np.array(S)
    
    theta_vector = cap(theta[0],0.,2*np.pi,1e-6)
    sintheta = np.sin(theta_vector)[:,None]
    y=np.cos(theta_vector)
    y = y - 1e-6 * (y==1.)
    
    allplm = np.stack( PlmSchmidt_d1_temp(l,y), axis=0 )[:,:,(l*(l+1))//2:(l*(l+1))//2+l+1]
    allplm = allplm[:,0,:]
    
    ms = np.arange(-l,l+1,1)
    mge0=ms[l:]
    ml0=ms[:l]

    Fr_mge0 = l*(l+1)/r**2 * S[l:] * allplm
    Fr_ml0  = l*(l+1)/r**2 * S[:l] * allplm[:,1:][:,::-1]

    return (Fr_mge0[None,:,:]*np.cos(mge0[None,None,:]*phi[:,:,None])).sum(axis=2) + (Fr_ml0[None,:,:]*np.sin(ml0[None,None,:]*phi[:,:,None])).sum(axis=2)

def scalar_sph_fixedrl(theta,phi,scalar,r,l):
    """ - theta, phi have shapes (nPhi,nTheta) (grid)
    - scalar has shapes (2l+1,) (components of degree l and all orders)
    - l is int (degree)
    - r is float (radius)
    """
    scalar = np.array(scalar)
    
    theta_vector = cap(theta[0],0.,2*np.pi,1e-6)
    sintheta = np.sin(theta_vector)[:,None]
    y=np.cos(theta_vector)
    y = y - 1e-6 * (y==1.)
    
    allplm = np.stack( PlmSchmidt_d1_temp(l,y), axis=0 )[:,:,(l*(l+1))//2:(l*(l+1))//2+l+1]
    allplm = allplm[:,0,:]
    
    ms = np.arange(-l,l+1,1)
    mge0=ms[l:]
    ml0=ms[:l]

    F_mge0 = scalar[l:] * allplm
    F_ml0  = scalar[:l] * allplm[:,1:][:,::-1]

    return (F_mge0[None,:,:]*np.cos(mge0[None,None,:]*phi[:,:,None])).sum(axis=2) + (F_ml0[None,:,:]*np.sin(ml0[None,None,:]*phi[:,:,None])).sum(axis=2)

### Fixed r, all l, all m ###
###############################
def torpol_to_thetaphi_sph_fixedr(theta,phi,T,dSdr,r):
    """ Takes the spherical harmonics of the toroidal|d(poloidal)/dr component of a given vector at a given radius and returns radial component as a function of theta,phi 
    - theta, phi have shapes (nPhi,nTheta) (grid)
    - T,dSdR are dataarrays with all desired degrees and orders
    - r is float (radius)
    """
    Ftheta=0;Fphi=0
    for l in np.array(T.degree):
        print(l, end=' ')
        Fthetalm,Fphilm = torpol_to_thetaphi_sph_fixedrl(theta,phi,T.sel(degree=l,order=slice(-l,l)),dSdr.sel(degree=l,order=slice(-l,l)),r,l)
        Ftheta+=Fthetalm
        Fphi+=Fphilm
    return Ftheta,Fphi

def pol_to_r_sph_fixedr(theta,phi,S,r):
    """ Takes the spherical harmonics of the poloidal component of a given vector at a given radius and returns radial component as a function of theta,phi
    - theta, phi have shapes (nPhi,nTheta) (grid)
    - S is a dataarray with all desired degrees and orders
    - r is float (radius)
    """
    Fr=0
    for l in np.array(S.degree):
        print(l, end=' ')
        Fr+=pol_to_r_sph_fixedrl(theta,phi,S.sel(degree=l,order=slice(-l,l)),r,l)
    return Fr

def scalar_sph_fixedr(theta,phi,scalar,r):
    """ Takes the spherical harmonics of a given scalar at a given radius and returns the scalar as a function of theta,phi
    - theta, phi have shapes (nPhi,nTheta) (grid)
    - S is a dataarray with all desired degrees and orders
    - r is float (radius)
    """
    F=0
    for l in np.array(scalar.degree):
        print(l, end=' ')
        F+=scalar_sph_fixedrl(theta,phi,scalar.sel(degree=l,order=slice(-l,l)),r,l)
    return F

### Fixed r, all l, fixed m ###
###############################
def torpol_to_thetaphi_sph_fixedrm(theta,phi,T,dSdr,r,m):
    """ - theta, phi have shapes (nPhi,nTheta) (grid)
    - T,dSdR are dataarrays with all desired degrees
    - r is float (radius)
    - m is int (order)
    """
    ls = np.array(T.degree)
    T,dSdr = np.array(T),np.array(dSdr)

    theta_vector = cap(theta[0],0.,2*np.pi,1e-6)
    sintheta = np.sin(theta_vector)[:,None]
    y=np.cos(theta_vector)
    y = y - 1e-6 * (y==1.)
    
    m_indices = [(l*(l+1))//2 +abs(m) for l in ls]
    
    allplm = np.stack( PlmSchmidt_d1_temp(ls[-1],y), axis=0 )[:,:,m_indices]
    alldplmdy = allplm[:,1,:]
    alldplmdtheta = - sintheta * alldplmdy
    allplm = allplm[:,0,:]   # shape = (nTheta,nl)

    if m>=0:
        Ftheta_sin = -1/(r*sintheta) * m * T * allplm  # shape = (nTheta,nl)
        Ftheta_cos = 1/r * dSdr * alldplmdtheta        
        Fphi_sin = -1/(r*sintheta) * m * dSdr * allplm
        Fphi_cos = -1/r * T * alldplmdtheta
    else :
        Ftheta_cos_ml0 = 1/(r*sintheta) * m * T* allplm
        Ftheta_sin_ml0 = 1/r * dSdr * alldplmdtheta
        Fphi_cos = 1/(r*sintheta) * ml0 * dSdr * allplm
        Fphi_sin = -1/r * T * alldplmdtheta        
        
    Ftheta = Ftheta_cos[None,:,:]*np.cos(m*phi[:,:,None]) + Ftheta_sin[None,:,:]*np.sin(m*phi[:,:,None])  # shape = (nPhi,nTheta,nl)
    Fphi = Fphi_cos[None,:,:]*np.cos(m*phi[:,:,None]) + Fphi_sin[None,:,:]*np.sin(m*phi[:,:,None])

    return Ftheta.sum(axis=2),Fphi.sum(axis=2)

def pol_to_r_sph_fixedrm(theta,phi,S,r,m):
    """ - theta, phi have shapes (nPhi,nTheta) (grid)
    - S is a dataarray with all desired degrees
    - r is float (radius)
    - m is int (order)
    """
    ls = np.array(S.degree)
    S = np.array(S)

    theta_vector = cap(theta[0],0.,2*np.pi,1e-6)
    sintheta = np.sin(theta_vector)[:,None]
    y=np.cos(theta_vector)
    y = y - 1e-6 * (y==1.)
    
    m_indices = [(l*(l+1))//2 +abs(m) for l in ls]
    
    allplm = np.stack( PlmSchmidt_d1_temp(ls[-1],y), axis=0 )[:,:,m_indices]
    alldplmdy = allplm[:,1,:]
    alldplmdtheta = - sintheta * alldplmdy
    allplm = allplm[:,0,:]   # shape = (nTheta,nl)

    if m>=0:
        Fr_cos = ls*(ls+1)/r**2 * S * allplm  # shape = (nTheta,nl)
        Fr_sin = 0.*allplm # just so it has the right shape
    else :
        Fr_cos = 0.*allplm
        Fr_sin = ls*(ls+1)/r**2 * S * allplm 

    Fr = Fr_cos[None,:,:]*np.cos(m*phi[:,:,None]) + Fr_sin[None,:,:]*np.sin(m*phi[:,:,None])  # shape = (nPhi,nTheta,nl)
    return Fr.sum(axis=2)

def scalar_sph_fixedrm(theta,phi,scalar,m):
    ls = np.array(scalar.degree)
    scalar = np.array(scalar)

    theta_vector = cap(theta[0],0.,2*np.pi,1e-6)
    sintheta = np.sin(theta_vector)[:,None]
    y=np.cos(theta_vector)
    y = y - 1e-6 * (y==1.)
    
    m_indices = [(l*(l+1))//2 +abs(m) for l in ls]
    
    allplm = np.stack( PlmSchmidt_d1_temp(ls[-1],y), axis=0 )[:,:,m_indices]
    alldplmdy = allplm[:,1,:]
    alldplmdtheta = - sintheta * alldplmdy
    allplm = allplm[:,0,:]   # shape = (nTheta,nl)

    if m>=0:
        F_cos = scalar * allplm  # shape = (nTheta,nl)
        F_sin = 0.*allplm # just so it has the right shape
    else :
        F_cos = 0.*allplm
        F_sin = scalar * allplm 

    F = F_cos[None,:,:]*np.cos(m*phi[:,:,None]) + F_sin[None,:,:]*np.sin(m*phi[:,:,None])  # shape = (nPhi,nTheta,nl)
    return F.sum(axis=2)

######################################################################################
########  Torpol / spherical harmonics to F_tilde(y) (complex representation)  #######
######################################################################################

def torpol_to_thetaphi_ycomplex_fixedrlm(y,T,dSdr,r,l,m):
    T,dSdr = float(T),float(dSdr)
    sintheta = np.sqrt(1-y**2)
    
    PlmdPlmdy = PlmSchmidt_d1_vec(l,abs(m),y)
    Plm,dPlmdy = PlmdPlmdy[:,0],PlmdPlmdy[:,1]
    dPlmdtheta = -sintheta * dPlmdy
    
    if m >=0 :
        Ftheta_sin = -1/(r*sintheta) * m * T * Plm #theta component that is multiplied by sin(m*phi)
        Ftheta_cos = 1/r * dSdr * dPlmdtheta       #theta component that is multiplied by cos(m*phi)
        
        Fphi_sin = -1/(r*sintheta) * m * dSdr * Plm #phi component that is multiplied by sin(m*phi)
        Fphi_cos = -T/r * dPlmdtheta                #phi component that is multiplied by cos(m*phi)
    
    if m < 0 :
        Ftheta_cos = 1/(r*sintheta) * m * T * Plm 
        Ftheta_sin = 1/r * dSdr * dPlmdtheta     
        
        Fphi_cos = 1/(r*sintheta) * m * dSdr * Plm
        Fphi_sin = -T/r * dPlmdtheta               
    
    return Ftheta_cos-1j*Ftheta_sin, Fphi_cos-1j*Fphi_sin

def torpol_to_thetaphi_ycomplex_fixedm(y,T,dSdr,m):
    ls = np.array(T.degree)
    rs = np.array(T.radius)[:,None,None]
    T,dSdr = np.array(T)[:,None,:],np.array(dSdr)[:,None,:]

    y = y - 1e-6 * (y==1.)
    sintheta = np.sqrt(1-y**2)[None,:,None]
    
    m_indices = [(l*(l+1))//2 +abs(m) for l in ls]
    
    allplm = np.stack( PlmSchmidt_d1_temp(ls[-1],y), axis=0 )[:,:,m_indices]
    alldplmdy = allplm[:,1,:]
    alldplmdtheta = - sintheta * alldplmdy[None,:,:]
    allplm = allplm[:,0,:][None,:,:]   # shape = (1,nTheta,nl)

    if m>=0:
        Ftheta_sin = -1/(rs*sintheta) * m * T * allplm  # shape = (nr,nTheta,nl)
        Ftheta_cos = 1/rs * dSdr * alldplmdtheta        
        Fphi_sin = -1/(rs*sintheta) * m * dSdr * allplm
        Fphi_cos = -1/rs * T * alldplmdtheta
    else :
        Ftheta_cos_ml0 = 1/(rs*sintheta) * m * T* allplm
        Ftheta_sin_ml0 = 1/rs * dSdr * alldplmdtheta
        Fphi_cos = 1/(rs*sintheta) * ml0 * dSdr * allplm
        Fphi_sin = -1/rs * T * alldplmdtheta
        
    return Ftheta_cos.sum(axis=2)-1j*Ftheta_sin.sum(axis=2), Fphi_cos.sum(axis=2)-1j*Fphi_sin.sum(axis=2)
    
def pol_to_r_ycomplex_fixedm(y,S,m):
    ls = np.array(S.degree)
    rs = np.array(S.radius)[:,None,None]
    S = np.array(S)[:,None,:]

    y = y - 1e-6 * (y==1.)
    sintheta = np.sqrt(1-y**2)[None,:,None]
    
    m_indices = [(l*(l+1))//2 +abs(m) for l in ls]
    
    allplm = np.stack( PlmSchmidt_d1_temp(ls[-1],y), axis=0 )[:,:,m_indices]
    alldplmdy = allplm[:,1,:]
    alldplmdtheta = - sintheta * alldplmdy[None,:,:]
    allplm = allplm[:,0,:][None,:,:]   # shape = (1,nTheta,nl)
    
    if m>=0:
        Fr_cos = ls*(ls+1)/rs**2 * S * allplm  # shape = (nr,nTheta,nl)
        Fr_sin = 0.*S  #just so it has the right shape
    else :
        Fr_cos = 0.*S
        Fr_sin = ls*(ls+1)/rs**2 * S * allplm  # shape = (nr,nTheta,nl)      
    return Fr_cos.sum(axis=2)-1j*Fr_sin.sum(axis=2)

def scalar_ycomplex_fixedm(y,scalar,m):
    ls = np.array(scalar.degree)
    scalar = np.array(scalar)[:,None,:]

    y = y - 1e-6 * (y==1.)
    sintheta = np.sqrt(1-y**2)[None,:,None]
    
    m_indices = [(l*(l+1))//2 +abs(m) for l in ls]
    
    allplm = np.stack( PlmSchmidt_d1_temp(ls[-1],y), axis=0 )[:,:,m_indices]
    alldplmdy = allplm[:,1,:]
    alldplmdtheta = - sintheta * alldplmdy[None,:,:]
    allplm = allplm[:,0,:][None,:,:]   # shape = (1,nTheta,nl)
    
    if m>=0:
        F_cos = scalar * allplm  # shape = (nr,nTheta,nl)
        F_sin = 0.*allplm  #just so it has the right shape
    else :
        F_cos = 0.*allplm
        F_sin = scalar * allplm  # shape = (nr,nTheta,nl)      
    return F_cos.sum(axis=2)-1j*F_sin.sum(axis=2)





###########################################
########   Unforced   Wave model   ########
###########################################
from finitediff import get_weights
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

def secondDerivative(y):
    """Returns a triadiagonal matrix with diagonal elements -2/dy^2, and sup/subdiagonal elements 1/dy^2"""
    n = len(y)
    dy = y[1]-y[0]
    return (1/dy**2)*sps.diags([1, -2, 1], [-1, 0, 1], shape=(n,n)).toarray()

def firstDerivative(y):
    """Returns a differenciation matrix with second order accuracy :
        - triadiagonal matrix with diagonal elements 0, and sup/subdiagonal elements +/- 1/(2*dy)
        - exception for the bounds where a second order one-sided scheme is used
    """
    n = len(y)
    dy = y[1]-y[0]
    D1 = (1/(2*dy))*sps.diags([-1, 0., 1], [-1, 0, 1], shape=(n,n)).toarray()
    D1[0,:3] = np.array([-1.5,2.,-0.5])/dy
    D1[-1,-3:] = np.array([0.5,-2.,1.5])/dy
    
    return D1

def sety(ngrid,spacing):
    """Sets the grid (ngrid points between -1 and 1, without the boundaries)"""
    if spacing=='linear':
        return np.linspace(-1.,1.,ngrid+2)[1:-1] #exclude the -1 and 1 bounds to avoid divisions by 0
    elif spacing=='cos':
        return np.cos(np.linspace(0.,np.pi,ngrid+2)[1:-1][::-1]) #exclude the -1 and 1 bounds to avoid divisions by 0
    else:
        raise ValueError("spacing must be one of ('linear','cos')")

def setyA(ngrid,m,M,spacing='linear'):
    """ Sets matrix A of the eigenvalue problem (A - C*In)x = 0"""
    y  = sety(ngrid,spacing)
    if spacing =='linear':
        D2 = secondDerivative(y)
    else:
        D2 = make_D_fornberg(y,2,npoints=7)
    L1 = -np.dot(np.diag(M*(1-y**2)/(y**2)), D2) + np.diag(M*(m**2-1)/(y**2 * (1-y**2)))

    L2 = -np.diag(m/(y**2))

    return y,np.block([[np.zeros((ngrid,ngrid)),np.eye(ngrid)],[L1,L2]])

def modesy(A,C):
    """Solves the linear eigenvalue problem (A - C*In)x = 0 and returns the corresponding b_theta''(y)"""
    w,v = spl.eig(A)
    i = np.argmin((np.real(w) - C)**2)

    return w[i],v[:,i]

###########################################
##########   Forced Wave model   ##########
###########################################


def responseForcing(ngrid,C,m,M,G,Ftheta,Flambda,adaptchi=True,Cref=0,chiref=0,spacing='linear'):
    ngrid=len(y)
    if adaptchi :
        if Cref == 0:
            raise ValueError("Please provide non-zero reference C / reference chi if using adaptive dissipation")
        chinew = 1 + (chiref-1)*Cref/C
        M = M*chiref/chinew
        ngrid=len(y)
        A=setA(ngrid,m,M,spacing)
        A = sps.csc_matrix(A)
    
    if spacing =='linear':
        D1 = firstDerivative(y)
    else:
        D1 = make_D_fornberg(y,1,npoints=5)
    
    forcingtheta = G*m**2/(M*(1-y**2)) * Ftheta
    forcinglambda = -1j* ( (C*G/M + 2*G/m)*y*Flambda 
                          + m*G/M *(1-y**2)**(-0.5) * np.dot(D1,Flambda*(1-y**2)**(0.5)) 
                         )
    rhs = np.concatenate([np.zeros(ngrid),forcingtheta+forcinglambda])

    return spsl.spsolve(A-C*sps.eye(2*ngrid),rhs)[:ngrid]/(1-y**2)

def plotResponses(ax,ngrid,m,M,G,Ftheta,Flambda,desc,adaptchi=True,Cref=0,chiref=0,spacing='linear',bounds=[-20,20]):
    y = sety(ngrid,spacing)
    A = sps.csc_matrix(setA(ngrid,m,M,spacing))
    lbound,rbound=bounds
    Cs = np.linspace(lbound,rbound,1000)
    
    t=time.time()
    responses = np.array([responseForcing(y,A,C,m,M,G,Ftheta,Flambda,adaptchi,Cref,chiref,spacing) for C in Cs])
    print(time.time()-t)
    
    #erase peaked response around frequency=0 if chi is adapted
    if adaptchi:
        responses[np.abs(Cs)<0.1]=0.
    
    lbl = "%s, adaptive chi = %r"%(desc,adaptchi)
    
    ax[0].set_title("Real part of (A-CIn)^-1 * F",fontsize=15)
    ax[0].plot(Cs,spl.norm(np.real(responses),axis=1),label=lbl)
    #ax[0].plot(Cs,np.sum(np.real(responses),axis=1),label=lbl)
    
    ax[1].set_title("Imaginary part of (A-CIn)^-1 * F",fontsize=15)
    ax[1].plot(Cs,spl.norm(np.imag(responses),axis=1),label=lbl)
    
    for a in ax:
        a.set_xlabel("C (nondimensional)")
        a.set_ylabel("Amplitude of the wave response")
        a.legend()
        
    w,_ = spl.eig(A.todense())
    for wi in w:
        if np.real(wi)>lbound and np.real(wi)<rbound:
            for a in ax:
                a.axvline(np.real(wi),color='k',linewidth=0.2)
         
                









                
                
######################################################################################
######  Other vizualization tools  #######
######################################################################################

def torpol_to_theta_sph_fixedr2(theta,phi,T,dSdr,r):
    """tried to vectorize all at once to avoid looping over ls but proved inefficient"""
    ls = np.array(T.degree)
    T,dSdr = np.array(T),np.array(dSdr)

    lmax = len(T)
    Tmge0 = np.hstack([T[l-ls[0]][~np.isnan(T[l-ls[0]])][l:] for l in ls])
    Tmle0 = np.hstack([np.insert(T[l-ls[0]][~np.isnan(T[l-ls[0]])][:l][::-1],0,0.) for l in ls])
    dSdrmge0 = np.hstack([dSdr[l-ls[0]][~np.isnan(dSdr[l-ls[0]])][l:] for l in ls])
    dSdrmle0 = np.hstack([np.insert(dSdr[l-ls[0]][~np.isnan(dSdr[l-ls[0]])][:l][::-1],0,0.) for l in ls])    
    
    theta_vector = cap(theta[0],0.,2*np.pi,1e-6)
    sintheta = np.sin(theta_vector)[:,None]
    y=np.cos(theta_vector)
    y = y - 1e-6 * (y==1.)
    
    allplm = np.stack( PlmSchmidt_d1_temp(ls[-1],y), axis=0 )[:,:,(ls[0]*(ls[0]+1))//2:]
    alldplmdy = allplm[:,1,:]
    alldplmdtheta = - sintheta * alldplmdy
    allplm = allplm[:,0,:]
    
    mge0 = np.hstack([np.arange(0,l+1,1) for l in ls])
    mle0 = np.hstack([np.arange(0,-l-1,-1) for l in ls])
    ms = np.hstack([mle0,mge0])

    Ftheta_sin_mge0 = -1/(r*sintheta) * mge0 * Tmge0 * allplm
    Ftheta_cos_mge0 = 1/r * dSdrmge0 * alldplmdtheta
    Ftheta_cos_ml0 = 1/(r*sintheta) * mle0 * Tmle0* allplm
    Ftheta_sin_ml0 = 1/r * dSdrmle0 * alldplmdtheta
        
    Ftheta_cos=np.hstack([Ftheta_cos_ml0,Ftheta_cos_mge0])
    Ftheta_sin=np.hstack([Ftheta_sin_ml0,Ftheta_sin_mge0])

    Ftheta = np.vstack([Ftheta_cos[None,:,:]]*len(theta[:,0]))*np.cos(ms[None,None,:]*phi[:,:,None]) + np.vstack([Ftheta_sin[None,:,:]]*len(theta[:,0]))*np.sin(ms[None,None,:]*phi[:,:,None])
    Ftheta = Ftheta.sum(axis=2)
        
    return Ftheta

def pol_to_r_sph_fixedr2(theta,phi,S,r):
    """tried to vectorize all at once to avoid looping over ls but proved inefficient"""
    ls = np.array(S.degree)
    S = np.array(S)

    lmax = len(S)
    Smge0 = np.hstack([S[l-ls[0]][~np.isnan(S[l-ls[0]])][l:] for l in ls])
    Smle0 = np.hstack([np.insert(S[l-ls[0]][~np.isnan(S[l-ls[0]])][:l][::-1],0,0.) for l in ls])
    
    theta_vector = cap(theta[0],0.,2*np.pi,1e-6)
    sintheta = np.sin(theta_vector)[:,None]
    y=np.cos(theta_vector)
    y = y - 1e-6 * (y==1.)
    
    allplm = np.stack( PlmSchmidt_d1_temp(ls[-1],y), axis=0 )[:,:,(ls[0]*(ls[0]+1))//2:]
    allplm = allplm[:,0,:]
    
    mge0 = np.hstack([np.arange(0,l+1,1) for l in ls])
    mle0 = np.hstack([np.arange(0,-l-1,-1) for l in ls])
    lls  = np.hstack([np.linspace(l,l,l+1) for l in ls])
        
    Fr_mge0 = lls*(lls+1)/r**2 * Smge0 * allplm
    Fr_mle0 = lls*(lls+1)/r**2 * Smle0 * allplm

    return (Fr_mge0[None,:,:]*np.cos(mge0[None,None,:]*phi[:,:,None])).sum(axis=2) + (Fr_mle0[None,:,:]*np.sin(mle0[None,None,:]*phi[:,:,None])).sum(axis=2)











