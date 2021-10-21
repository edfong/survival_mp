import jax.numpy as jnp
from jax import custom_jvp,jit
from jax.scipy.stats import norm,t
from jax.scipy.special import ndtri

### Functions for Clayton copula ###
#Calculate bivariate normal copula log H_uv and log c_uv
@jit
def clayton_logdistribution_logdensity(u,v,a):
    #clip to prevent 0s and 1s in CDF
    eps = 1e-6
    u = jnp.clip(u,eps,1-eps) 
    v = jnp.clip(v,eps,1-eps)
    
    #initialize useful terms
    u_ = 1-u
    v_ = 1-v
    k = (a+1)/a
    denom = u_**(-1/a) + v_**(-1/a)-1

    #compute log cop density and conditional cdf
    logcop_dens = jnp.log(k) - k*(jnp.log(u_) + jnp.log(v_)) -(a+2)*jnp.log(denom)
    log1cop_dist = jnp.clip(-k*jnp.log(v_) - (a+1)*jnp.log(denom),jnp.log(eps),jnp.log(1-eps))
    logcop_dist = jnp.log1p(-jnp.exp(log1cop_dist))

    return logcop_dist,logcop_dens
### ###


### Functions for Gaussian Copula ###
@custom_jvp #forward diff (define jvp for faster derivatives)
def ndtri_(u):
    return ndtri(u)
@ndtri_.defjvp
def f_jvp(primals, tangents):
    u, = primals
    u_dot, = tangents
    primal_out = ndtri_(u)
    tangent_out = (1/norm.pdf(primal_out))*u_dot
    return primal_out, tangent_out
ndtri_ = jit(ndtri_)

@custom_jvp #forward diff (define jvp for faster derivatives)
def norm_logcdf(z):
    return norm.logcdf(z)
@norm_logcdf.defjvp
def f_jvp(primals, tangents):
    z, = primals
    z_dot, = tangents
    primal_out = norm_logcdf(z)
    tangent_out = jnp.exp(norm.logpdf(z)-primal_out)*z_dot
    return primal_out, tangent_out
norm_logcdf = jit(norm_logcdf)

@jit #Calculate normal copula cdf and logpdf for f32
def norm_copula_logdistribution_logdensity(u,v,rho):
    #clip to prevent 0s and 1s in CDF, needed for numerical stability in high d.
    eps = 1e-6
    u = jnp.clip(u,eps,1-eps) 
    v = jnp.clip(v,eps,1-eps)
    
    #for reverse mode
    pu = ndtri_(u)
    pv = ndtri_(v)
    z = (pu - rho*pv)/jnp.sqrt(1- rho**2)

    logcop_dist = norm_logcdf(z)
    logcop_dist = jnp.clip(logcop_dist,jnp.log(eps),jnp.log(1-eps))
    logcop_dens = -0.5*jnp.log(1-rho**2) + (0.5/(1-rho**2))*(-(rho**2)*(pu**2 + pv**2)+ 2*rho*pu*pv)

    return logcop_dist,logcop_dens