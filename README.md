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

We recommend to do this in a clean virtual env:
```
python3 -m venv ~/virtualenvs/survival_copula
source ~/virtualenvs/survival_copula/bin/activate
```
where ```~/virtualenvs``` can be your preferred directory.

# Structure
All the main functions are in ```copula_survival_functions.py```.

# Experiments
Experiment run scripts are kept in the ```run_expts``` folder, including notebooks to plot. 
# survival_mp
