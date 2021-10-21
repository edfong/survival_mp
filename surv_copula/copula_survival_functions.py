import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap,jacfwd,jacrev,random,remat,value_and_grad
from jax.scipy.special import ndtri,erfc,logsumexp,betainc
from jax.scipy.stats import norm,t
from jax.lax import fori_loop,scan
from jax.ops import index, index_add, index_update
from jax.experimental import loops
from functools import partial

from .utils.bivariate_copula import clayton_logdistribution_logdensity

### Utility functions ###
@jit
def resample_is(log_w,i): #Multinomial resampling (systematic tedious to implement in Jax)
    key = random.PRNGKey(i)
    B = jnp.shape(log_w)[0]
    p = jnp.exp(log_w - logsumexp(log_w))
    
    #Measure ESS
    ESS = 1/np.sum(p**2)
    I_resample = jnp.where(ESS < 0.5*B, x = True, y = False)
    
    #Update or keep old indices/weights depending on ESS
    ind_old = jnp.arange(B)
    ind_new = I_resample * random.choice(key,B,shape = (B,), p = p) + (1-I_resample)*ind_old
    log_w_new = I_resample* jnp.log(np.ones(B)) + (1-I_resample)*log_w
        
    return log_w_new,ind_new,ESS

#Initialize marginal p_0, Lomax(a,b)
def init_marginals_single(y_test,a):
    #a = 1
    b = 1 #does this need adapting?

    #R+ CASE    
    logpdf_init= -(a+1)*jnp.log(1+y_test/b) +jnp.log(a) - jnp.log(b)
    logcdf_init= jnp.log1p(-(1+y_test/b)**(-a))

    return  logcdf_init,logpdf_init
init_marginals = jit(vmap(init_marginals_single,(0,None)))


#Compute copula update for a single data point
def update_copula_single(logcdf,logpdf,u,v,logalpha,a): 

    logcop_distribution,logcop_dens = clayton_logdistribution_logdensity(u,v,a)
    log1alpha = jnp.log1p(-jnp.exp(logalpha))

    logcdf = jnp.logaddexp((log1alpha + logcdf),(logalpha + logcop_distribution))
    logpdf= jnp.logaddexp(log1alpha, (logalpha+logcop_dens))+logpdf     

    return logcdf,logpdf

update_copula = jit(vmap(update_copula_single,(0,0,0,None,None,None))) #map over multiple y-values
update_copula_B = jit(vmap(update_copula,(0,0,0,0,None,None)))  #map over multiple IS samples
### ###

@jit
def update_pn(carry,i):
    vn,log_w,ESS,particle_ind,a_rand,delta,logcdf_yn,logpdf_yn,preq_loglik,a = carry
 
    logalpha = jnp.log(2.- (1/(i+1))) - jnp.log(i+2) #original (2-1/n)(n+1)^{-1} sequence
    #logalpha = 0 #uncomment for parametric

    u = jnp.exp(logcdf_yn)
    v = jnp.exp(logcdf_yn[:,i])
    v_y = delta[i]*v + (1-delta[i])*(v + a_rand[:,i]*(1-v)) #If censored, draw y~predictive
    vn = index_update(vn,index[:,i],v_y) #remember history of vn = P_{i-1}(y_i) (which may be simulated)

    #compute prequential likelihood
    log1p_c = jnp.log1p(-v) #IS weight increment log (1-P_{i-1}(c_i))
    logpdf_preq = logpdf_yn[:,i]
    logz_ratio = logsumexp(delta[i]*logpdf_preq + (1-delta[i])*log1p_c + log_w) - logsumexp(log_w)
    preq_loglik = index_update(preq_loglik, i,logz_ratio)

    #Update IS weights
    log_w =log_w+delta[i]*logpdf_preq + (1-delta[i])*log1p_c #add IS weight

    #update
    logcdf_yn,logpdf_yn= update_copula_B(logcdf_yn,logpdf_yn,u,v_y,logalpha,a)
    #a =  a + 1 #uncomment for parametric

    #Compute ESS and resample
    log_w, ind_new, ESS_new = resample_is(log_w,i)
    ESS = index_update(ESS,i,ESS_new)
    logcdf_yn = logcdf_yn[ind_new]
    logpdf_yn = logpdf_yn[ind_new]
    vn = vn[ind_new]

    #track particle history
    particle_ind = index_update(particle_ind,i+1, particle_ind[i,ind_new])

    carry = vn,log_w,ESS,particle_ind,a_rand,delta,logcdf_yn,logpdf_yn,preq_loglik,a
    return carry,i
    

#Scan over y_{1:n}
@jit
def update_pn_scan(carry,rng):
    return scan(update_pn,carry,rng)

#Compute P_{i-1}(y_i) and P_{i-1}(c_i), t should be ordered
@partial(jit,static_argnums = (4))
def update_pn_loop(key,t,delta,a,B):
    n = jnp.shape(t)[0]
    
    #Initialize terms
    vn = jnp.zeros((B,n)) #conditional cdf history of yn
    preq_loglik = np.zeros(n)
    log_w =jnp.zeros(B) #log importance weights
    ESS = jnp.zeros(n) #track ESS
    particle_ind = jnp.zeros((n+1,B)) #track initial particle uniqueness
    particle_ind = index_update(particle_ind,0, jnp.arange(B))
    
    #Initialize cdf/pdf (for each particle)
    logcdf_yn, logpdf_yn= init_marginals(t,a)
    logcdf_yn = np.repeat(logcdf_yn.reshape(1,-1),B,axis = 0)
    logpdf_yn = np.repeat(logpdf_yn.reshape(1,-1),B,axis = 0)

    #draw random uniform rvs
    key, subkey = random.split(key) #split key
    a_rand = random.uniform(subkey,shape = (B,n))
    
    #carry out for loop
    carry = vn,log_w,ESS,particle_ind,a_rand,delta,logcdf_yn,logpdf_yn,preq_loglik,a
    rng = jnp.arange(n)
    carry,rng = update_pn_scan(carry,rng)

    vn,log_w,ESS,particle_ind,a_rand,delta,logcdf_yn,logpdf_yn,preq_loglik,*_ = carry

    return vn,log_w,ESS,particle_ind,logcdf_yn,logpdf_yn,preq_loglik


### Optimizing preq loglik ##
@partial(jit,static_argnums = (4))
def nll(log_a,key,t,delta,B):
    a = jnp.exp(log_a[0])
    *_,preq_loglik= update_pn_loop(key,t,delta,a,B)
    return -jnp.sum(preq_loglik)

grad_nll = jit(jacfwd(nll),static_argnums =(4))

### Test point ###
@jit
def update_ptest(carry,i):
    vn,logcdf_ytest,logpdf_ytest,a = carry

    logalpha = jnp.log(2.- (1/(i+1))) - jnp.log(i+2)

    u = jnp.exp(logcdf_ytest)
    v = vn[:,i]

    logcdf_ytest,logpdf_ytest=update_copula_B(logcdf_ytest,logpdf_ytest,u,v,logalpha,a)

    carry = vn,logcdf_ytest,logpdf_ytest,a
    return carry,i

@jit
def update_ptest_scan(carry,rng):
    return scan(update_ptest,carry,rng)

@jit
def update_ptest_loop(vn,y_test,a):
    B = jnp.shape(vn)[0]
    n = jnp.shape(vn)[1]

    #Initialize cdf/pdf (for each particle)
    logcdf_ytest, logpdf_ytest= init_marginals(y_test,a)
    logcdf_ytest = np.repeat(logcdf_ytest.reshape(1,-1),B,axis = 0)
    logpdf_ytest = np.repeat(logpdf_ytest.reshape(1,-1),B,axis = 0)

    carry = vn,logcdf_ytest,logpdf_ytest,a
    rng = jnp.arange(n)
    carry,rng = update_ptest_scan(carry,rng)
    vn,logcdf_ytest,logpdf_ytest,a = carry

    return logcdf_ytest,logpdf_ytest

