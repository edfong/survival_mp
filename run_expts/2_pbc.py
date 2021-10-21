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

#Import data
data = pd.read_csv('./data/pbc.csv')
t = np.array(data['t'])
delta = np.array(data['delta'])
delta[delta ==1.] = 0
delta[delta==2.] = 1
trt = np.array(data['trt'])

#Split into treatments (filtering NA)
t1 = t[trt == 1.]
delta1 = delta[trt==1.]

t2 = t[trt == 2.]
delta2 = delta[trt==2.]

#Normalize
#Treatment
scale1 = np.sum(t1)/np.sum(delta1)
t1_norm = t1/scale1

#Placebo
scale2 = np.sum(t2)/np.sum(delta2)
t2_norm = t2/scale2


#Initialize plot and sample number
B = 2000 #number of posterior samples
T = 2000 #number of forward samples
key = random.PRNGKey(101)
dy  = 0.01
y_plot = np.arange(dy,1.5,dy)

np.savetxt("data/pbc_y_plot.csv",y_plot,delimiter = ',')


#NONPARAMETRIC PREDICTIVE SMC#

## TREATMENT ##
#Specify a_grid to choose a
a_grid = np.array([0.5,0.6,0.7,0.8,0.9])
cop_surv_obj  = fit_copula_survival(t1_norm,delta1, B,a_grid = a_grid)
print('Nonparametric a is {}'.format(cop_surv_obj.a_opt))

#Compute predictive density
logcdf_av, logpdf_av = predict_copula_survival(cop_surv_obj,y_plot)

#Predictive resample
logcdf_pr, logpdf_pr = predictive_resample_survival(cop_surv_obj,y_plot, T_fwdsamples = T)

#Assessing convergence
_,_,pdiff,cdiff = check_convergence_pr(cop_surv_obj,y_plot,10,25000)

#Save all files
jnp.save("data/pbc1_scale",scale1)
jnp.save('plot_files/pbc1_ESS_copula',cop_surv_obj.ESS)
jnp.save('plot_files/pbc1_particle_ind_copula',cop_surv_obj.particle_ind)
jnp.save('plot_files/pbc1_logpdf_av_copula',logpdf_av)
jnp.save('plot_files/pbc1_logcdf_av_copula',logcdf_av)
jnp.save('plot_files/pbc1_logpdf_samp_copula',logpdf_pr)
jnp.save('plot_files/pbc1_logcdf_samp_copula',logcdf_pr)
jnp.save('plot_files/pbc1_pdiff_copula',pdiff)
jnp.save('plot_files/pbc1_cdiff_copula',cdiff)
jnp.save('plot_files/pbc1_a_copula', cop_surv_obj.a_opt)


## PLACEBO ##
#Specify a_grid to choose a
a_grid = np.array([1.1,1.2,1.3,1.4,1.5])
cop_surv_obj  = fit_copula_survival(t2_norm,delta2, B,a_grid = a_grid)
print('Nonparametric a is {}'.format(cop_surv_obj.a_opt))

#Compute predictive density
logcdf_av, logpdf_av = predict_copula_survival(cop_surv_obj,y_plot)

#Predictive resample
logcdf_pr, logpdf_pr = predictive_resample_survival(cop_surv_obj,y_plot, T_fwdsamples = T)

#Assessing convergence
_,_,pdiff,cdiff = check_convergence_pr(cop_surv_obj,y_plot,10,25000)

#Save all files
jnp.save("data/pbc2_scale",scale2)
jnp.save('plot_files/pbc2_ESS_copula',cop_surv_obj.ESS)
jnp.save('plot_files/pbc2_particle_ind_copula',cop_surv_obj.particle_ind)
jnp.save('plot_files/pbc2_logpdf_av_copula',logpdf_av)
jnp.save('plot_files/pbc2_logcdf_av_copula',logcdf_av)
jnp.save('plot_files/pbc2_logpdf_samp_copula',logpdf_pr)
jnp.save('plot_files/pbc2_logcdf_samp_copula',logcdf_pr)
jnp.save('plot_files/pbc2_pdiff_copula',pdiff)
jnp.save('plot_files/pbc2_cdiff_copula',cdiff)
jnp.save('plot_files/pbc2_a_copula', cop_surv_obj.a_opt)
