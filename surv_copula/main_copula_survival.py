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
from .copula_survival_functions import nll,grad_nll,update_pn_loop,update_ptest_loop
from .sample_survival_functions import predictive_resample_loop, pr_loop_conv

### Fitting ###
#Compute overhead v_{1:n}, return fit copula object for prediction
def fit_copula_survival(t,delta,B,seed = 20,a_grid = None,B_opt = None):
    if B_opt is None:
        B_opt = B

    #Set seed for scipy
    np.random.seed(seed)

    #Generate random permutations
    key = PRNGKey(seed)
    key,*subkey = split(key,B +1) 

    #calculate a_opt
    #Compiling
    print('Compiling...')
    start = time.time()
    temp = nll(jnp.array([0.]),key,t,delta,B_opt).block_until_ready()
    if a_grid is None:
        temp = grad_nll(jnp.array([0.]),key,t,delta,B_opt).block_until_ready()
    temp =update_pn_loop(key,t,delta,1.,B)[0].block_until_ready()
    end = time.time()
    print('Compilation time: {}s'.format(round(end-start, 3)))

    print('Optimizing...')
    start = time.time()
    if a_grid is None:

        #Initialize parameter and put on correct scale to lie in >0
        log_a_init  = fit_parametric_a0(t,delta)

        #Carry out gradient descent
        opt = minimize(fun = nll,args = (key,t,delta,B_opt), x0= log_a_init,jac =grad_nll,method = 'SLSQP')

        #check optimization succeeded
        if opt.success == False:
            print('Optimization failed')

        #unscale hyperparameter
        a_opt = jnp.exp(opt.x)[0]

    else:
        #evaluate on grid
        n_a = np.shape(a_grid)[0]
        nll_a = np.zeros(n_a)
        for i in range(n_a):
            nll_a[i] = nll(np.array([np.log(a_grid[i])]),key,t,delta,B_opt)

        a_opt = a_grid[np.argmin(nll_a)]

    end = time.time()

    print('Optimization time: {}s'.format(round(end-start, 3)))
        
    print('Fitting...')
    start = time.time()
    vn,log_w,ESS,particle_ind,_,_,preq_loglik= update_pn_loop(key,t,delta,a_opt,B)
    temp = vn.block_until_ready()
    end = time.time()
    print('Fit time: {}s'.format(round(end-start, 3)))

    copula_survival_obj = namedtuple('copula_survival_obj',['vn','log_w','a_opt','preq_loglik', 'ESS','particle_ind'])
    return copula_survival_obj(vn,log_w,a_opt,preq_loglik, ESS, particle_ind)

#Helper function to estimate parametric a_0
def fit_parametric_a0(t,delta):
    b0 = 1
    n_uncen = np.sum(delta)
    opt = minimize(fun = par_nll,args = (b0,n_uncen,t), x0= 0.,jac =grad_par_nll,method = 'SLSQP')
    log_a_init  = opt.x
    return log_a_init


#Predict on test data using copula object
def predict_copula_survival(copula_survival_obj,y_test):
    print('Predicting...')
    start = time.time()
    logcdf,logpdf = update_ptest_loop(copula_survival_obj.vn,y_test,copula_survival_obj.a_opt)
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
def predictive_resample_survival(copula_survival_obj,y_test, T_fwdsamples = 2000, seed = 100, resample = True):
    #Fit cdf/pdf
    logcdf,logpdf = update_ptest_loop(copula_survival_obj.vn,y_test,copula_survival_obj.a_opt)

    #Initialize random seeds
    key = PRNGKey(seed)
    key,*subkey = split(key)

    #Forward sample
    print('Predictive resampling...')
    start = time.time()
    B = jnp.shape(copula_survival_obj.vn)[0]
    n = jnp.shape(copula_survival_obj.vn)[1] #get original data size
    logcdf_pr, logpdf_pr = predictive_resample_loop(key,logcdf,logpdf,copula_survival_obj.a_opt,n,T_fwdsamples)
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
def check_convergence_pr(copula_survival_obj,y_test,B,T_fwdsamples = 10000, seed = 100):
    #Fit cdf/pdf
    logcdf,logpdf = update_ptest_loop(copula_survival_obj.vn,y_test,copula_survival_obj.a_opt)

    #Initialize random seeds
    key = PRNGKey(seed)
    key,*subkey = split(key)

    #Forward sample
    print('Predictive resampling...')
    start = time.time()
    n = jnp.shape(copula_survival_obj.vn)[1] #get original data size
    logcdf_pr, logpdf_pr,pdiff,cdiff = pr_loop_conv(key,logcdf[0:B],logpdf[0:B],copula_survival_obj.a_opt,n,T_fwdsamples)
    logcdf_pr = logcdf_pr.block_until_ready() #for accurate timing
    end = time.time()
    print('Predictive resampling time: {}s'.format(round(end-start, 3)))
    return logcdf_pr,logpdf_pr,pdiff,cdiff
### ###




