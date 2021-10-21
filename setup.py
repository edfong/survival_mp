from setuptools import setup,find_packages
#May need to install Pystan separately with pip
setup(name='surv_copula',
      version='0.1.0',
      description='Survival Analysis with Copulas',
      author='Edwin Fong',
      author_email='edwin.fong@stats.ox.ac.uk',
      license='BSD 3-Clause',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn',
          'pandas',
          'matplotlib',
          'seaborn',
          'joblib',
          'tqdm',
          'jax',
          'jaxlib',
          'pydataset',
          'xlrd',
          'lifelines',
          'jupyter'
      ],
      include_package_data=True,
      python_requires='>=3.7'
      )
