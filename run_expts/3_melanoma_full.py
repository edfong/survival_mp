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
from surv_copula.main_copula_survreg_gaussian import fit_copula_survival,predict_copula_survival,check_convergence_pr,predictive_resample_survival

#Import data
data = pd.read_csv('./data/melanoma.csv')
t = np.array(data['t'])
delta = np.array(data['delta'])
x = np.array(data['x'])

#Normalize
scale = (np.sum(t)/np.sum(delta))
t_norm = t/scale
mean_x = np.mean(x)
std_x = np.std(x)
x_norm =(x- mean_x)/std_x

#Randomize
np.random.seed(120)
n = np.shape(t_norm)[0]
ind = np.random.permutation(np.arange(n))
t_norm = t_norm[ind]
delta = delta[ind]
x_norm = x_norm[ind]


#Initialize plot and sample number
B = 2000 #number of posterior samples
T = 10000 #number of forward samples
key = random.PRNGKey(101)
y_plot = np.arange(0,np.max(t),100)/scale


#NONPARAMETRIC PREDICTIVE SMC#

## TREATMENT ##
#Specify a_grid to choose a
rho_grid = np.array([0.5,0.6,0.7,0.8,0.9])
rho_,rho_x_ = np.meshgrid(rho_grid,rho_grid)
hyperparam_grid = np.vstack([rho_.ravel(), rho_x_.ravel()]).transpose()

# #Pass grid of a_values
cop_surv_obj  = fit_copula_survival(t_norm,delta,x_norm, B,hyperparam_grid = hyperparam_grid)

#Gradient
#cop_surv_obj  = fit_copula_survival(t_norm,delta,x_norm, B)
print('Optimal rho is {}'.format(cop_surv_obj.rho_opt))
print('Optimal rho_x is {}'.format(cop_surv_obj.rho_x_opt))

#Compute predictive cdf for various x values for survival plot
for x_ in np.array([1.5,3.4,6.1]):
	x_norm_ = (x_- mean_x)/std_x
	x_plot = x_norm_*np.ones(np.shape(y_plot)[0])
	logcdf_av, logpdf_av = predict_copula_survival(cop_surv_obj,y_plot,x_plot)
	jnp.save('plot_files/melanoma_logpdf_av_copula_x{}'.format(x_),logpdf_av)
	jnp.save('plot_files/melanoma_logcdf_av_copula_x{}'.format(x_),logcdf_av)

#Predictive resample at x= 3.4
x_norm_ = (3.4- mean_x)/std_x
x_plot = x_norm_*np.ones(np.shape(y_plot)[0])
logcdf_pr, logpdf_pr = predictive_resample_survival(cop_surv_obj,y_plot,x_plot, T_fwdsamples = T)

#Assessing convergence
_,_,pdiff,cdiff = check_convergence_pr(cop_surv_obj,y_plot,x_plot,5,25000)

#Save all files
jnp.save('plot_files/melanoma_ESS_copula',cop_surv_obj.ESS)
jnp.save('plot_files/melanoma_particle_ind_copula',cop_surv_obj.particle_ind)
jnp.save('plot_files/melanoma_logpdf_samp_copula',logpdf_pr)
jnp.save('plot_files/melanoma_logcdf_samp_copula',logcdf_pr)
jnp.save('plot_files/melanoma_pdiff_copula',pdiff)
jnp.save('plot_files/melanoma_cdiff_copula',cdiff)
jnp.save('plot_files/melanoma_rho_copula', cop_surv_obj.rho_opt)
jnp.save('plot_files/melanoma_rhox_copula', cop_surv_obj.rho_x_opt)
jnp.save('plot_files/melanoma_y_plot',y_plot)

#compute cdf to obtain median function
dy_med  = 0.02
y_plot_med = np.arange(dy_med,1.5,dy_med)
n_x = 40
x_grid = np.linspace(np.min(x_norm), np.max(x_norm),n_x)
median_fun = np.zeros(n_x)
for i in range(n_x):
	x_plot = x_grid[i]*np.ones(np.shape(y_plot_med)[0])
	logcdf_av, logpdf_av = predict_copula_survival(cop_surv_obj,y_plot_med,x_plot)
	median_fun[i] = y_plot_med[np.argmin(np.abs(np.exp(logcdf_av) - 0.5))]

jnp.save('plot_files/melanoma_median_fun',median_fun)
jnp.save('plot_files/melanoma_x_grid',x_grid)