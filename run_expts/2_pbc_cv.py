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

#Initialize cv
rep_cv = 10
n1 = np.shape(t1)[0]
n_train1 = int(n1/2)
n_test1 = n1-n_train1

n2 = np.shape(t2)[0]
n_train2 = int(n2/2)
n_test2 = n2-n_train2

test_ll_cv1 = np.zeros(rep_cv)
test_ll_cv2 = np.zeros(rep_cv)

seed = 100

for i in tqdm(range(rep_cv)):
    #Train-test split and save for R
    train_ind1,test_ind1 = train_test_split(np.arange(n1),test_size = n_test1,train_size = n_train1,random_state = seed+i)
    train_ind2,test_ind2 = train_test_split(np.arange(n2),test_size = n_test2,train_size = n_train2,random_state = seed+i)

    t1_train = t1[train_ind1]
    delta1_train = delta1[train_ind1]

    t1_test = t1[test_ind1]
    delta1_test = delta1[test_ind1]

    #normalize
    scale1 = (np.sum(t1_train)/np.sum(delta1_train))
    t1_train = t1_train/scale1
    t1_test = t1_test/scale1

    #save for R
    np.savetxt("data/pbc_t1_train{}.csv".format(i),t1_train,delimiter = ',')
    np.savetxt("data/pbc_delta1_train{}.csv".format(i),delta1_train,delimiter = ',')
    np.savetxt("data/pbc_t1_test{}.csv".format(i),t1_test,delimiter = ',')
    np.savetxt("data/pbc_delta1_test{}.csv".format(i),delta1_test,delimiter = ',')


    t2_train = t2[train_ind2]
    delta2_train = delta2[train_ind2]

    t2_test = t2[test_ind2]
    delta2_test = delta2[test_ind2]

    #normalize
    scale2 = (np.sum(t2_train)/np.sum(delta2_train))
    t2_train = t2_train/scale2
    t2_test = t2_test/scale2

    #save for R
    np.savetxt("data/pbc_t2_train{}.csv".format(i),t2_train,delimiter = ',')
    np.savetxt("data/pbc_delta2_train{}.csv".format(i),delta2_train,delimiter = ',')
    np.savetxt("data/pbc_t2_test{}.csv".format(i),t2_test,delimiter = ',')
    np.savetxt("data/pbc_delta2_test{}.csv".format(i),delta2_test,delimiter = ',')


    #Initialize plot and sample number
    B = 2000 #number of posterior samples

    #NONPARAMETRIC PREDICTIVE SMC#

    ## TREATMENT ##
    #Specify a_grid to choose a
    a_grid = np.array([1.1,1.2,1.3,1.4,1.5])
    cop_surv_obj  = fit_copula_survival(t1_train,delta1_train, B,a_grid = a_grid)
    print('Nonparametric a is {}'.format(cop_surv_obj.a_opt))

    #Compute predictive density
    logcdf_av, logpdf_av = predict_copula_survival(cop_surv_obj,t1_test)
    test_ll = (delta1_test)*(logpdf_av) + (1-delta1_test)*(np.log1p(-np.exp(logcdf_av)))
    test_ll_cv1[i] = np.mean(test_ll)

    ## PLACEBO ##
    #Specify a_grid to choose a
    a_grid = np.array([1.1,1.2,1.3,1.4,1.5])
    cop_surv_obj  = fit_copula_survival(t2_train,delta2_train, B,a_grid = a_grid)
    print('Nonparametric a is {}'.format(cop_surv_obj.a_opt))

    #Compute predictive density
    logcdf_av, logpdf_av = predict_copula_survival(cop_surv_obj,t2_test)
    test_ll = (delta2_test)*(logpdf_av) + (1-delta2_test)*(np.log1p(-np.exp(logcdf_av)))
    test_ll_cv2[i] = np.mean(test_ll)

print("{} +- {}".format(np.mean(test_ll_cv1),np.std(test_ll_cv1)/np.sqrt(rep_cv)))
print("{} +- {}".format(np.mean(test_ll_cv2),np.std(test_ll_cv2)/np.sqrt(rep_cv)))

