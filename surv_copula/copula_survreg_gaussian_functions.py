import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap,jacfwd,jacrev,random,remat,value_and_grad
from jax.scipy.special import ndtri,erfc,logsumexp,betainc
from jax.scipy.stats import norm,t
from jax.lax import fori_loop,scan
from jax.ops import index, index_add, index_update
from jax.experimental import loops
from functools import partial

from .copula_survival_functions import resample_is
from .utils.bivariate_copula import norm_copula_logdistribution_logdensity


### Utility functions ###
#Initialize marginals
def init_marginals_single(y_test,rho):

    ##CONTINUOUS CASE
    #normal(0,1)
    mean0 = 0.
    std0 = jnp.sqrt(1/(1-rho))

    logcdf_init = norm.logcdf(jnp.log(y_test),loc = mean0,scale = std0)#marginal initial cdfs
    logpdf_init = norm.logpdf(jnp.log(y_test),loc = mean0,scale = std0)- jnp.log(y_test) #marginal initial pdfs

    return  logcdf_init,logpdf_init
init_marginals = jit(vmap(init_marginals_single,(0,None)))

#Compute log k_xx for a single data point
@jit
def calc_logkxx_single(x,x_new,rho_x):
    logk_xx = -0.5*jnp.sum(jnp.log(1-rho_x**2)) -jnp.sum((0.5/(1-rho_x**2))*(((rho_x**2)*(x**2 + x_new**2) - 2*rho_x*x*x_new)))
    return logk_xx
calc_logkxx = jit(vmap(calc_logkxx_single,(0,None,None)))
calc_logkxx_test = jit(vmap(calc_logkxx,(None,0,None)))
### ###

### ###
def update_copula_single(logcdf,logpdf,u,v,logalpha,rho): 
    logcop_distribution,logcop_dens = norm_copula_logdistribution_logdensity(u,v,rho)
    log1alpha = jnp.log1p(-jnp.exp(logalpha))
    logcdf = jnp.logaddexp((log1alpha + logcdf),(logalpha + logcop_distribution))
    logpdf = jnp.logaddexp(log1alpha, (logalpha+logcop_dens))+logpdf     
    return logcdf,logpdf

update_copula = jit(vmap(update_copula_single,(0,0,0,None,0,None))) #map over multiple y-values and x-values
update_copula_B = jit(vmap(update_copula,(0,0,0,0,None,None)))  #map over multiple IS samples

@jit
def update_pn(carry,i):
    vn,log_w,ESS,particle_ind,a_rand,delta,x,logcdf_yn,logpdf_yn,preq_loglik,rho,rho_x = carry
 
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

    #Compute new x
    x_new = x[i]
    logalpha = jnp.log(2.- (1/(i+1)))-jnp.log(i+2)
    #logalpha = 0 #uncomment for parametric

    #compute x rhos/alphas
    logk_xx = calc_logkxx(x,x_new,rho_x)
    logalphak_xx = logalpha + logk_xx
    log1alpha = jnp.log1p(-jnp.exp(logalpha))
    logalpha_x = (logalphak_xx) - (jnp.logaddexp(log1alpha,logalphak_xx)) #alpha*k_xx /(1-alpha + alpha*k_xx)

    #update
    logcdf_yn,logpdf_yn= update_copula_B(logcdf_yn,logpdf_yn,u,v_y,logalpha_x,rho)

    #Compute ESS and resample
    log_w, ind_new, ESS_new = resample_is(log_w,i)
    ESS = index_update(ESS,i,ESS_new)
    logcdf_yn = logcdf_yn[ind_new]
    logpdf_yn = logpdf_yn[ind_new]
    vn = vn[ind_new]

    #track particle history
    particle_ind = index_update(particle_ind,i+1, particle_ind[i,ind_new])

    carry = vn,log_w,ESS,particle_ind,a_rand,delta,x,logcdf_yn,logpdf_yn,preq_loglik,rho,rho_x
    return carry,i
    

#Scan over y_{1:n}
@jit
def update_pn_scan(carry,rng):
    return scan(update_pn,carry,rng)

#Compute P_{i-1}(y_i) and P_{i-1}(c_i), t should be ordered
@partial(jit,static_argnums = (6))
def update_pn_loop(key,t,delta,x,rho,rho_x,B):
    n = jnp.shape(t)[0]
    
    #Initialize terms
    vn = jnp.zeros((B,n)) #conditional cdf history of yn
    preq_loglik = np.zeros(n)
    log_w =jnp.zeros(B) #log importance weights
    ESS = jnp.zeros(n) #track ESS
    particle_ind = jnp.zeros((n+1,B)) #track initial particle uniqueness
    particle_ind = index_update(particle_ind,0, jnp.arange(B))
    
    #Initialize cdf/pdf (for each particle)
    logcdf_yn, logpdf_yn= init_marginals(t,rho)
    logcdf_yn = np.repeat(logcdf_yn.reshape(1,-1),B,axis = 0)
    logpdf_yn = np.repeat(logpdf_yn.reshape(1,-1),B,axis = 0)

    #draw random uniform rvs
    key, subkey = random.split(key) #split key
    a_rand = random.uniform(subkey,shape = (B,n))
    
    #carry out for loop
    carry = vn,log_w,ESS,particle_ind,a_rand,delta,x,logcdf_yn,logpdf_yn,preq_loglik,rho,rho_x
    rng = jnp.arange(n)
    carry,rng = update_pn_scan(carry,rng)

    vn,log_w,ESS,particle_ind,a_rand,delta,x,logcdf_yn,logpdf_yn,preq_loglik,*_ = carry

    return vn,log_w,ESS,particle_ind,logcdf_yn,logpdf_yn,preq_loglik

### Optimizing preq loglik ##
@partial(jit,static_argnums = (5))
def nll(log_hyperparam,key,t,delta,x,B):
    rho =  1/(1+jnp.exp(log_hyperparam[0]))
    rho_x = 1/(1+jnp.exp(log_hyperparam[1]))
    *_,preq_loglik= update_pn_loop(key,t,delta,x,rho,rho_x,B)
    return -jnp.sum(preq_loglik)

grad_nll = jit(jacfwd(nll),static_argnums =(5))

### Test point ###
@jit
def update_ptest(carry,i):
    vn,x,x_test,logcdf_ytest,logpdf_ytest,rho,rho_x = carry

    #Compute new x
    x_new = x[i]
    logalpha = jnp.log(2.- (1/(i+1)))-jnp.log(i+2)
    u = jnp.exp(logcdf_ytest)
    v = vn[:,i]

    #compute x rhos/alphas
    logk_xx = calc_logkxx(x_test,x_new,rho_x)
    logalphak_xx = logalpha + logk_xx
    log1alpha = jnp.log1p(-jnp.exp(logalpha))
    logalpha_x = (logalphak_xx) - (jnp.logaddexp(log1alpha,logalphak_xx)) #alpha*k_xx /(1-alpha + alpha*k_xx)

    #update
    logcdf_ytest,logpdf_ytest= update_copula_B(logcdf_ytest,logpdf_ytest,u,v,logalpha_x,rho)

    carry = vn,x,x_test,logcdf_ytest,logpdf_ytest,rho,rho_x
    return carry,i

@jit
def update_ptest_scan(carry,rng):
    return scan(update_ptest,carry,rng)

@jit
def update_ptest_loop(vn,x,y_test,x_test,rho,rho_x):
    B = jnp.shape(vn)[0]
    n = jnp.shape(vn)[1]

    #Initialize cdf/pdf (for each particle)
    logcdf_ytest, logpdf_ytest= init_marginals(y_test,rho)
    logcdf_ytest = np.repeat(logcdf_ytest.reshape(1,-1),B,axis = 0)
    logpdf_ytest = np.repeat(logpdf_ytest.reshape(1,-1),B,axis = 0)

    carry = vn,x,x_test,logcdf_ytest,logpdf_ytest,rho,rho_x
    rng = jnp.arange(n)
    carry,rng = update_ptest_scan(carry,rng)
    _,_,_,logcdf_ytest,logpdf_ytest,*_ = carry

    return logcdf_ytest,logpdf_ytest

