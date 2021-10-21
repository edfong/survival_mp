import numpy as np
import scipy as sp
import pandas as pd
from tqdm import tqdm
import copy
import time

from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random
from surv_copula.main_copula_survival import fit_copula_survival,fit_parametric_a0,\
                                                predict_copula_survival,check_convergence_pr,predictive_resample_survival
from surv_copula.parametric_survival_functions import pr_lomax_smc,pr_lomax_IS

#Generate data
np.random.seed(101)
n = 50
y = sp.stats.expon.rvs(size = n, scale = 1) #survival
c = sp.stats.expon.rvs(size= n,scale = 0.5) #censoring

t = copy.copy(y)
t[y>c] = c[y>c]
delta = np.ones(n,dtype = 'int')
delta[y>c] = 0
dy = 0.1
y_plot = np.arange(dy,10,dy)

#Print info
n_uncen = np.sum(delta)
n_cen = n - n_uncen
print("Fraction of censoring is {}".format(n_cen/n))

#Normalize
scale = np.sum(t)/n_uncen
t = t/scale

#Save for R
np.savetxt("data/sim_t.csv",t,delimiter = ',')
np.savetxt("data/sim_delta.csv",delta,delimiter = ',')
np.savetxt("data/sim_y_plot.csv",y_plot,delimiter = ',')
np.save("data/sim_scale",scale)

#Initialize plot and sample number
B = 2000 #number of posterior samples
T = 2000 #number of forward samples
key = random.PRNGKey(101)

#PARAMETRIC PREDICTIVE SMC#

#Estimate a_0 from parametric model to determine grid
a0 = jnp.exp(fit_parametric_a0(t,delta))
b0 = 1
print(r'Parametric a_0 is {}'.format(a0))

#IS
start = time.time()
a_samp_IS, b_samp_IS,log_w_IS = pr_lomax_IS(a0,b0,t,delta,key,B,T) #Naive IS
temp = b_samp_IS.block_until_ready()
end = time.time()
print('Parametric IS required {}s'.format(round(end-start, 3)))

#SMC
start = time.time()
a_samp_smc, b_samp_smc,log_w_smc,ESS_smc,particle_ind_smc, theta_hist_smc = pr_lomax_smc(a0,b0,t,delta,key,B,T) 
temp = b_samp_IS.block_until_ready()
end = time.time()
print('Parametric SMC required {}s'.format(round(end-start, 3)))

#Save all files
jnp.save('plot_files/sim_a_samp_smc',a_samp_smc)
jnp.save('plot_files/sim_b_samp_smc',b_samp_smc)
jnp.save('plot_files/sim_log_w_smc',log_w_smc)
jnp.save('plot_files/sim_ESS_smc',ESS_smc)
jnp.save('plot_files/sim_particle_ind_smc',particle_ind_smc)
jnp.save('plot_files/sim_theta_hist_smc',theta_hist_smc)

jnp.save('plot_files/sim_a_samp_IS',a_samp_IS)
jnp.save('plot_files/sim_b_samp_IS',b_samp_IS)
jnp.save('plot_files/sim_log_w_IS',log_w_IS)
jnp.save('plot_files/sim_a0', a0)

##

#SUPPLEMENTARY EXPERIMENTS#
#Order t and delta so uncensored before right-censored
t_ord = np.zeros(n)
t_ord[0:n_uncen] = t[delta == 1]
t_ord[n_uncen:] = t[delta==0]

delta_ord = np.zeros(n,dtype = 'int')
delta_ord[0:n_uncen] = 1

#Estimate a_0 from parametric model to determine grid
a0 = jnp.exp(fit_parametric_a0(t_ord,delta_ord))
b0 = 1
print(r'Parametric a_0 is {}'.format(a0))

#IS
start = time.time()
a_samp_IS, b_samp_IS,log_w_IS = pr_lomax_IS(a0,b0,t_ord,delta_ord,key,B,T) #Naive IS
temp = b_samp_IS.block_until_ready()
end = time.time()
print('Parametric IS required {}s'.format(round(end-start, 3)))

#Save all files
jnp.save('plot_files/sim_a_samp_IS_ord',a_samp_IS)
jnp.save('plot_files/sim_b_samp_IS_ord',b_samp_IS)
jnp.save('plot_files/sim_log_w_IS_ord',log_w_IS)
jnp.save('plot_files/sim_a0_ord', a0)
##
