Simple examples of how to use our formulas to obtain approximations of the security level and the LWE dimension of a set of parameters.  


```python
import numpy as np
import math
from estimator import *
```

## Security level


```python
params_lambda_red_usvp_U3 = [0.83354, 0.15494, 1.46982, 18.09877]
params_lambda_red_usvp_U2 = [0.44530, 1.48698, 0.95011, 11.21416]
```


```python
def lambda_red_usvp(n,lnq,params):
    return params[0]*math.log((params[1]*n)/lnq)*(n/lnq) + params[2]*math.log(n) + params[3]
```

### An example


```python
logqs = [16,18,19,25,27]
n = 2**10

params = params_lambda_red_usvp_U3

for logq in logqs:
    lnq = math.log(2**logq)
    est_f = round(lambda_red_usvp(n,lnq,params))
    FHEParam = LWE.Parameters(n, q=2**logq, Xs=ND.UniformMod(3), Xe=ND.DiscreteGaussian(stddev=3.19))
    primal_usvp_cost = LWE.primal_usvp(FHEParam, red_cost_model=RC.BDGL16)
    print('logq: ', logq, "est formula: ", est_f, "est: ", round(math.log(primal_usvp_cost['rop'],2)))
```

    logq:  16 est formula:  233 est:  231.0
    logq:  18 est formula:  202 est:  204.0
    logq:  19 est formula:  190 est:  193.0
    logq:  25 est formula:  137 est:  143.0
    logq:  27 est formula:  126 est:  132.0


## LWE dimension


```python
params_n_usvp_U3 = [-1.07304, 0.27831, 0.93120, 0.792882]
params_n_usvp_U2 = [-1.142080, 0.231197, 1.106616, -0.233138]

params_n_bdd_U3 = [2.36830289, -0.67630697, -4.10436958, -19.1104812]
params_n_bdd_U2 = [2.46304008, 3.42658016, -24.9248619, 128.041708]
```


```python
def n_usvp(l,lnq,params):
    return np.multiply(np.divide(l + params[0]*np.log(lnq), params[2]+ params[1]*np.log(l)) + params[3], lnq)

def n_bdd(l,lnq,params):
    return np.multiply(params[0]*np.divide(l,np.log(l)) + params[1]*np.log(lnq) + params[2],lnq) + params[3]
```

### An example


```python
logqs = [[27,37,45,54]]
sec = [[128,128,129,128]]

params_usvp = params_n_usvp_U2
params_bdd = params_n_bdd_U2

for i in range(len(sec)):
    print("\n")
    for j in range(len(sec[0])):
        logq = logqs[i][j]
        l = sec[i][j]
        lnq = math.log(2**logq)
        
        n_u = int(round(n_usvp(l,lnq,params_usvp)))
        n_b = int(round(n_bdd(l,lnq,params_bdd)))
        
        FHEParam = LWE.Parameters(n_u, q=2**logq, Xs=ND.UniformMod(2), Xe=ND.DiscreteGaussian(stddev=3.19))
        primal_usvp_cost = LWE.primal_usvp(FHEParam, red_cost_model=RC.BDGL16)
        FHEParam = LWE.Parameters(n_b, q=2**logq, Xs=ND.UniformMod(2), Xe=ND.DiscreteGaussian(stddev=3.19))
        primal_bdd_cost = LWE.primal_bdd(FHEParam, red_cost_model=RC.BDGL16)
        
        print("logq ", logq, "est input ", l, "n usvp", n_u, "n bdd", n_b, "est usvp", round(math.log(primal_usvp_cost['rop'],2)), "est bdd", round(math.log(primal_bdd_cost['rop'],2)))
```

    
    
    logq  27 est input  128 n usvp 1043 n bdd 1065 est usvp 130.0 est bdd 131.0
    logq  37 est input  128 n usvp 1425 n bdd 1440 est usvp 130.0 est bdd 129.0
    logq  45 est input  129 n usvp 1742 n bdd 1758 est usvp 131.0 est bdd 130.0
    logq  54 est input  128 n usvp 2072 n bdd 2092 est usvp 129.0 est bdd 130.0

