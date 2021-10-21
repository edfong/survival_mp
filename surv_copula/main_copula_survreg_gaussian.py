from scipy.optimize import minimize
from collections import namedtuple
import time
import numpy as np
from tqdm import tqdm

#import jax
import jax.numpy as jnp
from jax import vmap
from jax.random import permutation,PRNGKey,split
from jax.scipy.special import logsumexp

#import package functions
from .parametric_survival_functions import par_nll,grad_par_nll
from .copula_survreg_gaussian_functions import nll,grad_nll,update_pn_loop,update_ptest_loop
from .sample_survreg_gaussian_functions import predictive_resample_loop, pr_loop_conv

### Fitting ###
#Compute overhead v_{1:n}, return fit copula object for prediction
def fit_copula_survival(t,delta,x,B,seed = 20,hyperparam_grid = None,B_opt = None,rho_init = 0.9, rho_x_init = 0.9):
    if B_opt is None:
        B_opt = B

    #Set seed for scipy
    np.random.seed(seed)

    #Generate random permutations
    key = PRNGKey(seed)
    key,*subkey = split(key,B +1) 

    #calculate rho_opt
    #Compiling
    print('Compiling...')
    start = time.time()
    temp = nll(jnp.log(jnp.array([1/rho_init - 1,1/rho_x_init - 1])),key,t,delta,x,B_opt).block_until_ready()
    if hyperparam_grid is None:
        temp = grad_nll(jnp.log(jnp.array([1/rho_init - 1,1/rho_x_init - 1])),key,t,delta,x,B_opt).block_until_ready()
    temp =update_pn_loop(key,t,delta,x,rho_init,rho_x_init,B)[0].block_until_ready()
    end = time.time()
    print('Compilation time: {}s'.format(round(end-start, 3)))

    print('Optimizing...')
    start = time.time()
    if hyperparam_grid is None:

        #Initialize parameter and put on correct scale to lie in >0
        #log_hyperparam_init  = fit_parametric_a0(t,delta)
        log_hyperparam_init = jnp.log(jnp.array([1/rho_init - 1,1/rho_x_init - 1]))

        #Carry out gradient descent
        opt = minimize(fun = nll,args = (key,t,delta,x,B_opt), x0= log_hyperparam_init,jac =grad_nll,method = 'SLSQP')

        #check optimization succeeded
        if opt.success == False:
            print('Optimization failed')

        #unscale hyperparameter
        rho_opt = 1/(1+jnp.exp(opt.x[0]))
        rho_x_opt = 1/(1+jnp.exp(opt.x[1]))

    else:
        #evaluate on grid
        n_h = np.shape(hyperparam_grid)[0]
        nll_h = np.zeros(n_h)
        for i in range(n_h):
            rho,rho_x = hyperparam_grid[i]
            nll_h[i] = nll(jnp.log(np.array([1/rho - 1,1/rho_x - 1])),key,t,delta,x,B_opt)

        rho_opt,rho_x_opt = hyperparam_grid[np.argmin(nll_h)]

    end = time.time()

    print('Optimization time: {}s'.format(round(end-start, 3)))
        
    print('Fitting...')
    start = time.time()
    vn,log_w,ESS,particle_ind,_,_,preq_loglik= update_pn_loop(key,t,delta,x,rho_opt,rho_x_opt,B)
    temp = vn.block_until_ready()
    end = time.time()
    print('Fit time: {}s'.format(round(end-start, 3)))

    copula_survival_obj = namedtuple('copula_survival_obj',['vn','log_w','rho_opt','rho_x_opt','x','preq_loglik', 'ESS','particle_ind'])
    return copula_survival_obj(vn,log_w,rho_opt,rho_x_opt,x,preq_loglik, ESS, particle_ind)


#Predict on test data using copula object
def predict_copula_survival(copula_survival_obj,y_test,x_test):
    print('Predicting...')
    start = time.time()
    logcdf,logpdf = update_ptest_loop(copula_survival_obj.vn,copula_survival_obj.x,y_test,x_test,copula_survival_obj.rho_opt,copula_survival_obj.rho_x_opt)
    logcdf = logcdf.block_until_ready() #for accurate timing
    end = time.time()

    #Take IS average
    log_w = copula_survival_obj.log_w - logsumexp(copula_survival_obj.log_w)
    logcdf_av = logsumexp(log_w.reshape(-1,1)+ logcdf,axis = 0)
    logpdf_av = logsumexp(log_w.reshape(-1,1)+ logpdf,axis = 0)
    print('Prediction time: {}s'.format(round(end-start, 3)))
    return logcdf_av,logpdf_av

### Predictive Resampling ###
#Forward sampling without diagnostics for speed
def predictive_resample_survival(copula_survival_obj,y_test,x_test, T_fwdsamples = 2000, seed = 100, resample = True):
    #Fit cdf/pdf
    logcdf,logpdf = update_ptest_loop(copula_survival_obj.vn,copula_survival_obj.x,y_test,x_test,copula_survival_obj.rho_opt,copula_survival_obj.rho_x_opt)

    #Initialize random seeds
    key = PRNGKey(seed)
    key,*subkey = split(key)

    #Forward sample
    print('Predictive resampling...')
    start = time.time()
    B = jnp.shape(copula_survival_obj.vn)[0]
    n = jnp.shape(copula_survival_obj.vn)[1] #get original data size
    logcdf_pr, logpdf_pr = predictive_resample_loop(key,logcdf,logpdf,copula_survival_obj.x,x_test,copula_survival_obj.rho_opt,copula_survival_obj.rho_x_opt,n,T_fwdsamples)
    logcdf_pr = logcdf_pr.block_until_ready() #for accurate timing
    end = time.time()
    print('Predictive resampling time: {}s'.format(round(end-start, 3)))

    #Return IS samples
    log_w = copula_survival_obj.log_w - logsumexp(copula_survival_obj.log_w)
    if resample == True:
        w = jnp.exp(log_w)
        ind = np.random.choice(np.arange(B),size = B, p = w, replace = True)
        logcdf_pr = logcdf_pr[ind]
        logpdf_pr = logpdf_pr[ind]
        return logcdf_pr,logpdf_pr
    else:
        return logcdf_pr,logpdf_pr,log_w

#Check convergence by running 1 long forward sample chain
def check_convergence_pr(copula_survival_obj,y_test,x_test,B,T_fwdsamples = 10000, seed = 100):
    #Fit cdf/pdf
    logcdf,logpdf = update_ptest_loop(copula_survival_obj.vn,copula_survival_obj.x,y_test,x_test,copula_survival_obj.rho_opt,copula_survival_obj.rho_x_opt)

    #Initialize random seeds
    key = PRNGKey(seed)
    key,*subkey = split(key)

    #Forward sample
    print('Predictive resampling...')
    start = time.time()
    n = jnp.shape(copula_survival_obj.vn)[1] #get original data size
    logcdf_pr, logpdf_pr,pdiff,cdiff = pr_loop_conv(key,logcdf[0:B],logpdf[0:B],copula_survival_obj.x,x_test,copula_survival_obj.rho_opt,copula_survival_obj.rho_x_opt,n,T_fwdsamples)
    logcdf_pr = logcdf_pr.block_until_ready() #for accurate timing
    end = time.time()
    print('Predictive resampling time: {}s'.format(round(end-start, 3)))
    return logcdf_pr,logpdf_pr,pdiff,cdiff
### ###




