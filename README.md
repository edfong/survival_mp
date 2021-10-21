# survival_copula
Martingale posteriors for survival analysis

# Installation

To install the package, just run the following in the main folder:
```
python setup.py install
```
This may not work for newer Macs, in which case we recommend using ``pip`` instead in the main folder:
```
pip install .
```

We recommend creating a clean virtual environment before doing the above:
```
python3 -m venv ~/virtualenvs/survival_copula
source ~/virtualenvs/survival_copula/bin/activate
```
where ```~/virtualenvs``` can be your preferred directory. Please also make sure to have the latest version of `setuptools` or `pip` before installing.

Please check the [JAX](https://github.com/google/jax) page for CPU versus GPU usage and installation instructions. For the paper results, we use the CPU version for reproducibility as GPU calculations can be non-deterministic, and timing was carried out on the GPU version. For full reproducibility of the experiments in the paper, please use the versions `jax==0.2.21` and `jaxlib==0.1.71`. 

The version of R used is 4.1.1 for the MCMC examples. Please install the `dirichletprocess` package [here](https://cran.r-project.org/web/packages/dirichletprocess/index.html) and the `ddpanova` package [here](https://web.ma.utexas.edu/users/pmueller/prog.html).

# Structure
All the main functions are in ```surv_copula/copula_survival_functions.py``` for the exponential copula with no covariates, and in  ```surv_copula/copula_survreg_gaussian_functions.py``` for the lognormal copula with covariates.

# Experiments
Experiment run scripts are kept in the ```run_expts``` folder, including notebooks to plot. The scripts are prefixed based on the order in which the experiments appear in the paper, and running the Python scripts should involve entering the following in terminal when in the `run_expt` folder, for example:
```
python3 1_sim.py
```
The R scripts should be run (in Rstudio or terminal) after the respective Python scripts, as some datasets are simulated/split into the `run_expt/data` folder by the Python scripts. 

Outputs from the experiments are stored in `run_expt/plot_files`, which are then used by the the respective Jupyter notebook with the prefix to produce the plots in the paper and supplementary material. 

# Data
We have included the following datasets in `run_expts/data` for convenience:
- PBC (Dickson et al., 1989)
- Melanoma (Venables & Ripley, 2013)
- Kidney (Klein et al., 2012)
The processing script used is provided in `R/process_data.R`.

# References
Dickson, E. R., Grambsch, P. M., Fleming, T. R., Fisher, L. D., & Langworthy, A. (1989). Prognosis in primary biliary cirrhosis: model for decision making. Hepatology, 10(1), 1-7.

Venables, W. N., & Ripley, B. D. (2002). Modern applied statistics with S (Fourth) [ISBN 0-387-95457-0]. Springer. http://www.stats.ox.ac.uk/pub/MASS4

Klein, Moeschberger, & modifications by Jun Yan. (2012). KMsurv: Data sets from Klein and Moeschberger (1997), Survival analysis [R package version 0.1-5]. https://CRAN.R-project.org/package=KMsurv
