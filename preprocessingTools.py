import numpy as np
import pandas as pd
import xarray as xr

from pyshtools.legendre import PlmSchmidt_d1
PlmSchmidt_d1_temp=np.vectorize(PlmSchmidt_d1,otypes=[np.ndarray],excluded=[0])


def dat_to_xarray(file):
    """
    Transforms a Calypso output into an xarray.Dataset, which can then be saved to netCDF format
    args :
        - file : str, path to a calypso output file
    returns :
        - ds : xarray.Dataset, where dimensions have been automatically extracted
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
        ds = ds.assign_coords({'time':ds.time.isel({c:-1 for c in coords if c != 't_step'}, drop=True)})
    if 'radius_ID' in coords:
        ds = ds.assign_coords({'radius':ds.radius.isel({c:-1 for c in coords if c != 'radius_ID'}, drop=True)})
    return ds

def scalar_ygrid_fixedm(y,scalar,m):
    """Transforms a scalar variable from spherical harmonic space to spherical coordinate space, 
       or more precisely from (radius,degree) space to (radius,y=cos(latitude)) space, for a fixed order m.
       To recover the phi dependence, multiply the output by e^{i*m*phi}) and take the real part
    args :
        - y : numpy.ndarray, the y (=cos(latitude)) grid on which to compute the output
        - scalar : xr.Dataarray, the two-dimensional (radius & spherical harmonic degree) scalar variable.
        - m : the spherical harmonic order of the data
    returns :
        - a two-dimensional numpy.ndarray, dimensions are ordered as (radius,y)
    """
    ls = np.array(scalar.degree)
    scalar = np.array(scalar)[:,None,:]

    y = y - 1e-6 * (y==1.)
    sintheta = np.sqrt(1-y**2)[None,:,None]
    
    m_indices = [(l*(l+1))//2 +abs(m) for l in ls]
    
    allplm = np.stack( PlmSchmidt_d1_temp(ls[-1],y), axis=0 )[:,:,m_indices]
    alldplmdy = allplm[:,1,:]
    alldplmdtheta = - sintheta * alldplmdy[None,:,:]
    allplm = allplm[:,0,:][None,:,:]   # shape = (1,ny,nl)
    
    if m>=0:
        F_cos = scalar * allplm  # shape = (nr,ny,nl)
        F_sin = 0.*allplm  #just so it has the right shape
    else :
        F_cos = 0.*allplm
        F_sin = scalar * allplm  # shape = (nr,ny,nl)      
    return F_cos.sum(axis=2)-1j*F_sin.sum(axis=2)


def torpol_to_rthetaphi_ygrid_fixedm(y,T,S,dSdr,m):
    """Transforms a vector variable from toroidal/poloidal components in spherical harmonic space to spherical components in spherical coordinate space. 
       More precisely, this function goes from (radius,degree) space to (radius,y=cos(latitude)) space, for a fixed order m.
       To recover the phi dependence, multiply the outputs by e^{i*m*phi}) and take the real part.
    args :
        - y : numpy.ndarray, the y (=cos(latitude)) grid on which to compute the output
        - T : xr.Dataarray, toroidal component, two-dimensional (radius & spherical harmonic degree) 
        - S : xr.Dataarray, poloidal component, two-dimensional (radius & spherical harmonic degree) 
        - dSdr : xr.Dataarray, d(poloidal component)/dr, two-dimensional (radius & spherical harmonic degree) 
        - m : the spherical harmonic order of the data
    returns :
        - three two-dimensional numpy.ndarray (r, theta and phi vector components), dimensions are ordered as (radius,y)
    """
    ls = np.array(T.degree)
    rs = np.array(T.radius)[:,None,None]
    T,S,dSdr = np.array(T)[:,None,:],np.array(S)[:,None,:],np.array(dSdr)[:,None,:]

    y = y - 1e-6 * (y==1.)
    sintheta = np.sqrt(1-y**2)[None,:,None]
    
    m_indices = [(l*(l+1))//2 +abs(m) for l in ls]
    
    allplm = np.stack( PlmSchmidt_d1_temp(ls[-1],y), axis=0 )[:,:,m_indices]
    alldplmdy = allplm[:,1,:]
    alldplmdtheta = - sintheta * alldplmdy[None,:,:]
    allplm = allplm[:,0,:][None,:,:]   # shape = (1,ny,nl)

    if m>=0:
        Fr_sin = 0.*S  #just so it has the right shape
        Fr_cos = ls*(ls+1)/rs**2 * S * allplm  # shape = (nr,ny,nl)
        Ftheta_sin = -1/(rs*sintheta) * m * T * allplm 
        Ftheta_cos = 1/rs * dSdr * alldplmdtheta        
        Fphi_sin = -1/(rs*sintheta) * m * dSdr * allplm
        Fphi_cos = -1/rs * T * alldplmdtheta
    else :
        Fr_cos = 0.*S
        Fr_sin = ls*(ls+1)/rs**2 * S * allplm  # shape = (nr,nTheta,nl)   
        Ftheta_cos = 1/(rs*sintheta) * m * T* allplm
        Ftheta_sin = 1/rs * dSdr * alldplmdtheta
        Fphi_cos = 1/(rs*sintheta) * m * dSdr * allplm
        Fphi_sin = -1/rs * T * alldplmdtheta
        
    return Fr_cos.sum(axis=2)-1j*Fr_sin.sum(axis=2), Ftheta_cos.sum(axis=2)-1j*Ftheta_sin.sum(axis=2), Fphi_cos.sum(axis=2)-1j*Fphi_sin.sum(axis=2)

def interp_scalar(y, scalar,m):
    """Transforms a scalar variable from (time,radius,degree) space to (time,radius,y=cos(latitude)) space, for a fixed order m.
       To recover the phi dependence, multiply the output by e^{i*m*phi}) and take the real part
    args :
        - y : numpy.ndarray, the y (=cos(latitude)) grid on which to compute the output
        - scalar : xr.Dataarray, the two-dimensional (radius & spherical harmonic degree) scalar variable.
        - m : the spherical harmonic order of the data
    returns :
        - a xarray.Dataset, with dimensions time, radius and y. It contains two variables (real and imaginary part of the scalar).
    """
    scalar_y_ar=[]
    for i in scalar.t_step:
        scalar_y_ar.append(scalar_ygrid_fixedm(y,scalar.sel(t_step=i).squeeze(),m))
    scalar_y_ar=np.array(scalar_y_ar)
    
    scalar_y = xr.DataArray(scalar_y_ar, 
                                 coords= {
                                     "t_step":scalar.t_step,
                                     "radius_ID":scalar.radius_ID,
                                     "y":y,
                                     "time":("t_step",scalar.time),
                                     "radius":("radius_ID",scalar.radius),
                                 },
                                 dims=["t_step", "radius_ID","y"])
    scalar_y=xr.merge([np.real(scalar_y).rename("%s_real"%scalar.name),np.imag(scalar_y).rename("%s_imag"%scalar.name)])
    return scalar_y
    
    
def interp_vector(y,vector,m):
    """Transforms a vector variable from (time,radius,degree) space and toroidal/poloidal components to (time,radius,y=cos(latitude)) space and spherical components, for a fixed order m.
       To recover the phi dependence, multiply the output by e^{i*m*phi}) and take the real part
    args :
        - y : numpy.ndarray, the y (=cos(latitude)) grid on which to compute the output
        - vector : xr.Dataset, the three-dimensional (time,radius & spherical harmonic degree) vector variable with three components (toroidal, poloidal, d(poloidal)/dr.
        - m : the spherical harmonic order of the data
    returns :
        - a xarray.Dataset, with dimensions time, radius and y. It contains six variables (real and imaginary parts of the r,theta and phi components).
    """
    name = list(vector.variables)[-3][:-4]
    
    vector_y_r_ar=[]
    vector_y_theta_ar=[]
    vector_y_phi_ar=[]
    for i in vector.t_step:
        r,theta,phi=torpol_to_rthetaphi_ygrid_fixedm(y,vector[name+'_tor'].sel(t_step=i), vector[name+'_pol'].sel(t_step=i), vector[name+'_pol_dr'].sel(t_step=i).squeeze(), m)
        vector_y_r_ar.append(r)
        vector_y_theta_ar.append(theta)
        vector_y_phi_ar.append(phi)
    vector_y_r_ar    =np.array(vector_y_r_ar    )
    vector_y_theta_ar=np.array(vector_y_theta_ar)
    vector_y_phi_ar  =np.array(vector_y_phi_ar  )
    
    vector_y = xr.Dataset({name+'_r_real':(["t_step","radius_ID","y"],np.real(vector_y_r_ar)),
                           name+'_r_imag':(["t_step","radius_ID","y"],np.imag(vector_y_r_ar)),
                           name+'_theta_real':(["t_step","radius_ID","y"],np.real(vector_y_theta_ar)),
                           name+'_theta_imag':(["t_step","radius_ID","y"],np.imag(vector_y_theta_ar)),
                           name+'_phi_real':(["t_step","radius_ID","y"],np.real(vector_y_phi_ar)),
                           name+'_phi_imag':(["t_step","radius_ID","y"],np.imag(vector_y_phi_ar))
                           }, 
                            coords= {
                                "t_step":vector.t_step,
                                "radius_ID":vector.radius_ID,
                                "y":y,
                                "time":("t_step",vector.time.data),
                                "radius":("radius_ID",vector.radius.data),
                            })
    return vector_y