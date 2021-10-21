import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap,jacfwd,jacrev,random,remat,value_and_grad
from jax.scipy.special import logsumexp, gammaln
from jax.scipy.stats import norm,t
from jax.lax import fori_loop,scan
from jax.ops import index, index_add, index_update
from jax.experimental import loops
from functools import partial

from .utils.lomax import dlomax,plomax,invplomax,rlomax,rlomax_cens
from .copula_survival_functions import resample_is

## Update Functions (in parallel) ##
@jit 
def sim_update_lomax_cen_smc(carry,i):
    a,b,log_weightIS,ESS,particle_ind,t,delta,key,theta_hist = carry
    B = np.shape(a)[0]
    key, subkey = random.split(key)
    
    log_weightIS += delta[i]*jnp.log(dlomax(t[i],a,b)) + (1-delta[i])*jnp.log1p(-plomax(t[i],a,b))
    y_sim = rlomax_cens(B,t[i],a,b,subkey)
    a +=  1
    b += delta[i]*t[i] + (1-delta[i])*y_sim
    
    #Resample (comment for IS without resampling)
    key, subkey = random.split(key) #split key
    log_weightIS, ind,ESS_new = resample_is(log_weightIS,i)
    a = a[ind]
    b = b[ind]
    ESS= index_update(ESS,i,ESS_new)
    
    #track particle history and mean parameter history
    particle_ind = index_update(particle_ind,i+1, particle_ind[i,ind])
    theta_hist = index_update(theta_hist,i,b/(a-1) )

    carry= a,b,log_weightIS,ESS,particle_ind,t,delta,key,theta_hist
    return carry,i
 
@jit
def sim_update_lomax_cen_smc_scan(carry,rng):
    return scan(sim_update_lomax_cen_smc,carry,rng)    

@jit 
def sim_update_lomax(carry,i):
    a,b,key,theta_hist = carry
    B = np.shape(a)[0]
    key, subkey = random.split(key)
    y_sim = rlomax(B,a,b,subkey)
    a +=  1
    b += y_sim

    #track parameter history
    theta_hist = index_update(theta_hist,i,b/(a-1) )
    carry= a,b,key,theta_hist
    return carry,i

@jit
def sim_update_lomax_scan(carry,rng):
    return scan(sim_update_lomax,carry,rng)


## Main Function ##
@partial(jit,static_argnums = (5,6))
def pr_lomax_smc(a0,b0,t,delta,key,B,T):
    n = np.shape(t)[0]
    key, subkey = random.split(key)
    ESS = B*np.ones(n)
    particle_ind = jnp.zeros((n+1,B)) #track initial particle uniqueness
    particle_ind = index_update(particle_ind,0, jnp.arange(B))
    theta_hist = jnp.zeros((n+T,B))

    carry = a0*np.ones(B),b0*np.ones(B),np.zeros(B),ESS,particle_ind,t,delta,subkey,theta_hist
    rng = jnp.arange(n)
    carry,rng = sim_update_lomax_cen_smc_scan(carry,rng)
    a_samp,b_samp,log_weightIS,ESS,particle_ind,_,_,_,theta_hist = carry
    
    key, subkey = random.split(key)
    carry = a_samp,b_samp,subkey,theta_hist
    rng = jnp.arange(n,n+T)
    carry,rng = sim_update_lomax_scan(carry,rng)
    a_samp,b_samp,_,theta_hist = carry
    return a_samp,b_samp,log_weightIS,ESS,particle_ind,theta_hist


## Marginal Likelihood ##
def par_nll(log_a0,b0,n_uncen,t):
    a0 = jnp.exp(log_a0)
    log_ml = a0*jnp.log(b0)+ gammaln(n_uncen+a0)- gammaln(a0)- (n_uncen + a0)*jnp.log(b0+ jnp.sum(t))
    return -log_ml

grad_par_nll = jit(jacfwd(par_nll))

## Naive IS code ##
## Main Function ##
@partial(jit,static_argnums = (5,6))
def pr_lomax_IS(a0,b0,t,delta,key,B,T):
    n= np.shape(t)[0]
    key, subkey = random.split(key)
    ESS = B*np.ones(n)
    theta_hist = jnp.zeros((n+T,B)) #not used

    carry = a0*np.ones(B),b0*np.ones(B),np.zeros(B),t,delta,subkey
    rng = jnp.arange(n)
    carry,rng = sim_update_lomax_cen_scan(carry,rng)
    a_samp,b_samp,log_weightIS,*_ = carry
    
    key, subkey = random.split(key)
    carry = a_samp,b_samp,subkey,theta_hist
    rng = jnp.arange(n,n+T)
    carry,rng = sim_update_lomax_scan(carry,rng)
    a_samp,b_samp,*_ = carry
    return a_samp,b_samp,log_weightIS

## Update Functions (in parallel) ##
@jit 
def sim_update_lomax_cen(carry,i):
    a,b,log_weightIS,t,delta,key = carry
    B = np.shape(a)[0]
    key, subkey = random.split(key)
    
    log_weightIS += delta[i]*jnp.log(dlomax(t[i],a,b)) + (1-delta[i])*jnp.log1p(-plomax(t[i],a,b))
    y_sim = rlomax_cens(B,t[i],a,b,subkey)
    a +=  1
    b += delta[i]*t[i] + (1-delta[i])*y_sim
    
    carry= a,b,log_weightIS,t,delta,key
    return carry,i
 
@jit
def sim_update_lomax_cen_scan(carry,rng):
    return scan(sim_update_lomax_cen,carry,rng) 