import numpy as np
import scipy as sp
import pandas as pd
from tqdm import tqdm
import copy
import time
from sklearn.model_selection import train_test_split


from jax.config import config
#config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random
from surv_copula.main_copula_survreg_gaussian import fit_copula_survival,predict_copula_survival,check_convergence_pr


def main_copula_survreg(dataset):

	#Import data
	data = pd.read_csv("./data/{}.csv".format(dataset))
	t = np.array(data['t'])
	delta = np.array(data['delta'])
	x = np.array(data['x'])
	n = np.shape(t)[0]

	rep_cv = 10
	if dataset =="melanoma":
		n_train = int(n/2)
	elif dataset == "kidney":
		#n_train = 100
		n_train = int(n/2)

	n_test = n-n_train

	test_ll_cv = np.zeros(rep_cv)
	seed = 100

	for i in tqdm(range(rep_cv)):
		#Train-test split and save for R
		train_ind,test_ind = train_test_split(np.arange(n),test_size = n_test,train_size = n_train,random_state = seed+i)

		t_train = t[train_ind]
		delta_train = delta[train_ind]
		x_train = x[train_ind]

		t_test = t[test_ind]
		delta_test = delta[test_ind]
		x_test = x[test_ind]

		#normalize
		scale = (np.sum(t_train)/np.sum(delta_train))
		t_train = t_train/scale
		t_test = t_test/scale

		mean_x = np.mean(x_train)
		std_x = np.std(x_train)

		x_train = (x_train - mean_x)/std_x
		x_test = (x_test - mean_x)/std_x

		#save for R
		np.savetxt("data/{}_t_train{}.csv".format(dataset,i),t_train,delimiter = ',')
		np.savetxt("data/{}_delta_train{}.csv".format(dataset,i),delta_train,delimiter = ',')
		np.savetxt("data/{}_x_train{}.csv".format(dataset,i),x_train,delimiter = ',')
		np.savetxt("data/{}_t_test{}.csv".format(dataset,i),t_test,delimiter = ',')
		np.savetxt("data/{}_delta_test{}.csv".format(dataset,i),delta_test,delimiter = ',')
		np.savetxt("data/{}_x_test{}.csv".format(dataset,i),x_test,delimiter = ',')


		#Initialize plot and sample number
		B = 2000 #number of posterior samples

		rho_grid = np.array([0.5,0.6,0.7,0.8,0.9])
		rho_,rho_x_ = np.meshgrid(rho_grid,rho_grid)
		hyperparam_grid = np.vstack([rho_.ravel(), rho_x_.ravel()]).transpose()

		#Pass grid of a_values
		cop_surv_obj  = fit_copula_survival(t_train,delta_train,x_train, B,hyperparam_grid = hyperparam_grid)

		print('Optimal rho is {}'.format(cop_surv_obj.rho_opt))
		print('Optimal rho_x is {}'.format(cop_surv_obj.rho_x_opt))

		logcdf_av, logpdf_av = predict_copula_survival(cop_surv_obj,t_test,x_test)
		test_ll = (delta_test)*(logpdf_av) + (1-delta_test)*(np.log1p(-np.exp(logcdf_av)))
		print(np.mean(test_ll))
		test_ll_cv[i] = np.mean(test_ll)

	print("{} +- {}".format(np.mean(test_ll_cv),np.std(test_ll_cv)/np.sqrt(rep_cv)))


main_copula_survreg("kidney")
main_copula_survreg("melanoma")
