import jax.numpy as jnp
from jax import jit
from jax import random
from functools import partial

### Functions for parametric Example ###
## Lomax Functions ##
@jit
def dlomax(x,a,b):
    return ((a/b)*(1+x/b)**(-(a+1))) #pdf at x
@jit
def plomax(x,a,b):
    return (1-(1+x/b)**(-a)) #cdf at x
@jit
def invplomax(p,a,b):
    return((((1-p)**(-1/a)) -1)*b) #inverse cdf of p

@partial(jit,static_argnums = (0,))
def rlomax(n,a,b,key): #generate n lomax r.v.
    u = random.uniform(shape = (n,),key= key)
    return(invplomax(u,a,b))

@partial(jit,static_argnums = (0,))
def rlomax_cens(n,c,a,b,key): #generate n lomax r.v. left-truncated at c
    pc = plomax(c,a,b)
    u = random.uniform(shape = (n,),key = key)*(1-pc)+ pc
    return (invplomax(u,a,b))