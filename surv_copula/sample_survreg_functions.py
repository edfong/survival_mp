import numpy as np
import scipy as sp
from functools import partial

#import jax functions
import jax.numpy as jnp
from jax import grad,value_and_grad, jit, vmap,jacfwd,jacrev,random
from jax.scipy.stats import norm
from jax.lax import fori_loop
from jax.ops import index_update

### Predictive resampling functions ###
from .copula_survreg_functions import update_copula_B, update_ptest_scan, calc_logkxx

#### Main function ####
# Loop through forward sampling; generate uniform random variables, then use p(y) update from surv
@partial(jit,static_argnums = (7,8))
def predictive_resample_loop(key,logcdf,logpdf,x,x_test,a,rho_x,n,T):
    B = np.shape(logcdf)[0]
    n = jnp.shape(logcdf)[1]

    #generate uniform random numbers
    key, subkey = random.split(key) #split key
    a_rand = random.uniform(subkey,shape = (B,T))

    #Draw random x_samp from BB
    key, subkey = random.split(key) #split key
    n = jnp.shape(x)[0]
    w = random.dirichlet(subkey, jnp.ones(n)) #single set of dirichlet weights
    key, subkey = random.split(key) #split key
    ind_new = random.choice(key,a = jnp.arange(n),p = w,shape = (1,T))[0]
    x_new = x[ind_new]

    #Append a_rand to empty vn (for correct array size)
    vT = jnp.concatenate((jnp.zeros((B,n)),a_rand),axis = 1)
    x_samp = jnp.concatenate((x,x_new),axis = 0)

    #run forward loop
    inputs = vT,x_samp,x_test,logcdf,logpdf,a,rho_x
    rng = jnp.arange(n,n+T)
    outputs,rng = update_ptest_scan(inputs,rng)
    vT,x_samp,x_test,logcdf,logpdf,a,rho_x = outputs

    return logcdf,logpdf
#### ####

#### Convergence checks ####

# Update p(y) in forward sampling, while keeping a track of change in p(y) for convergence check
def pr_1step_conv(i,inputs):  #t = n+i
    logcdf,logpdf,x_samp,x_test,logcdf_init,logpdf_init,pdiff,cdiff,a,rho_x,n,a_rand = inputs #a is d-dimensional uniform rv
    n_test = jnp.shape(logcdf)[1]

    #update pdf/cdf
    logalpha = jnp.log(2- (1/(n+i+1)))-jnp.log(n+i+2)
    x_new = x_samp[i]
    u = jnp.exp(logcdf)
    v = a_rand[:,i] #cdf of rv is uniformly distributed

    #compute x rhos/alphas
    logk_xx = calc_logkxx(x_test,x_new,rho_x)
    logalphak_xx = logalpha + logk_xx
    log1alpha = jnp.log1p(-jnp.exp(logalpha))
    logalpha_x = (logalphak_xx) - (jnp.logaddexp(log1alpha,logalphak_xx)) #alpha*k_xx /(1-alpha + alpha*k_xx)

    logcdf_new,logpdf_new= update_copula_B(logcdf,logpdf,u,v,logalpha_x,a)

    #joint density
    pdiff = index_update(pdiff,i,jnp.mean(jnp.abs(jnp.exp(logpdf_new)- jnp.exp(logpdf_init)),axis = 1)) #mean density diff from initial
    cdiff = index_update(cdiff,i,jnp.mean(jnp.abs(jnp.exp(logcdf_new)- jnp.exp(logcdf_init)),axis = 1)) #mean cdf diff from initial (only univariate)

    outputs = logcdf_new,logpdf_new,x_samp,x_test,logcdf_init,logpdf_init,pdiff,cdiff,a,rho_x,n,a_rand 
    return outputs

#Loop through forward sampling, starting with average p_n
@partial(jit,static_argnums = (7,8))
def pr_loop_conv(key,logcdf,logpdf,x,x_test,a,rho_x,n,T):
    B = jnp.shape(logcdf)[0]
    n = jnp.shape(logcdf)[1]

    #generate random numbers
    key, subkey = random.split(key) #split key
    a_rand = random.uniform(subkey,shape = (B,T))

    #Draw random x_samp from BB
    key, subkey = random.split(key) #split key
    n = jnp.shape(x)[0]
    w = random.dirichlet(subkey, jnp.ones(n)) #single set of dirichlet weights
    key, subkey = random.split(key) #split key
    ind_new = random.choice(key,a = jnp.arange(n),p = w,shape = (1,T))[0]
    x_new = x[ind_new]

    #Append x_new to x (for correct array size)
    x_samp = jnp.concatenate((x,x_new),axis = 0)

    #Track difference
    pdiff = jnp.zeros((T,B))
    cdiff = jnp.zeros((T,B))

    inputs = logcdf,logpdf,x_samp,x_test,logcdf,logpdf,pdiff,cdiff,a,rho_x,n,a_rand 

    #run loop
    outputs = fori_loop(0,T,pr_1step_conv,inputs)
    logcdf,logpdf,x_samp,x_test,logcdf_init,logpdf_init,pdiff,cdiff,*_ = outputs

    return logcdf,logpdf,pdiff,cdiff
#### ####
### ###



