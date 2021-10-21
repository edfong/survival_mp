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
from .copula_survival_functions import update_copula_B, update_ptest_scan

#### Main function ####
# Loop through forward sampling; generate uniform random variables, then use p(y) update from surv
@partial(jit,static_argnums = (4,5))
def predictive_resample_loop(key,logcdf,logpdf,a,n,T):
    B = np.shape(logcdf)[0]
    n = jnp.shape(logcdf)[1]

    #generate uniform random numbers
    key, subkey = random.split(key) #split key
    a_rand = random.uniform(subkey,shape = (B,T))

    #Append a_rand to empty vn (for correct array size)
    vT = jnp.concatenate((jnp.zeros((B,n)),a_rand),axis = 1)

    #run forward loop
    inputs = vT,logcdf,logpdf,a
    rng = jnp.arange(n,n+T)
    outputs,rng = update_ptest_scan(inputs,rng)
    vT,logcdf,logpdf,a = outputs

    return logcdf,logpdf
#### ####

#### Convergence checks ####

# Update p(y) in forward sampling, while keeping a track of change in p(y) for convergence check
def pr_1step_conv(i,inputs):  #t = n+i
    logcdf,logpdf,logcdf_init,logpdf_init,pdiff,cdiff,a,n,a_rand = inputs #a is d-dimensional uniform rv
    n_test = jnp.shape(logcdf)[1]

    #update pdf/cdf
    logalpha = jnp.log(2- (1/(n+i+1)))-jnp.log(n+i+2)

    u = jnp.exp(logcdf)
    v = a_rand[:,i] #cdf of rv is uniformly distributed

    logcdf_new,logpdf_new= update_copula_B(logcdf,logpdf,u,v,logalpha,a)

    #joint density
    pdiff = index_update(pdiff,i,jnp.mean(jnp.abs(jnp.exp(logpdf_new)- jnp.exp(logpdf_init)),axis = 1)) #mean density diff from initial
    cdiff = index_update(cdiff,i,jnp.mean(jnp.abs(jnp.exp(logcdf_new)- jnp.exp(logcdf_init)),axis = 1)) #mean cdf diff from initial (only univariate)

    outputs = logcdf_new,logpdf_new,logcdf_init,logpdf_init,pdiff,cdiff,a,n,a_rand 
    return outputs

#Loop through forward sampling, starting with average p_n
@partial(jit,static_argnums = (4,5))
def pr_loop_conv(key,logcdf,logpdf,a,n,T):
    B = jnp.shape(logcdf)[0]
    n = jnp.shape(logcdf)[1]

    #generate random numbers
    key, subkey = random.split(key) #split key
    a_rand = random.uniform(subkey,shape = (B,T))

    #Track difference
    pdiff = jnp.zeros((T,B))
    cdiff = jnp.zeros((T,B))

    inputs = logcdf,logpdf,logcdf,logpdf,pdiff,cdiff,a,n,a_rand 

    #run loop
    outputs = fori_loop(0,T,pr_1step_conv,inputs)
    logcdf,logpdf,logcdf_init,logpdf_init,pdiff,cdiff,a,n,a_rand = outputs

    return logcdf,logpdf,pdiff,cdiff
#### ####
### ###



