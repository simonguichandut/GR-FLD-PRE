import sys
sys.path.append(".")

## Paczynski and Anderson gave a simple formula for static envelopes when L\approx LEdd
## Following their derivations but keeping x=L/LEdd a free parameter, the result depends on the hypergeometric function (see our Appendix D)

import numpy as np
from scipy.special import hyp2f1
from scipy.optimize import fsolve

# Neutron star parameters
M = 1.4*2e33 # g
R = 12e5 # cm

# Constants
G = 6.6726e-8
kB = 1.380658e-16
arad = 7.5657e-15
c = 2.99792458e10
mp = 1.67e-24

# Opacity formula : kappa=kappa0 / (1 + (T/T0)^alpha)
T0 = 4.5e8 # K
alpha = 0.86

# To solve, we need to know the temperature at the base (rb=R). We use the BC that y=yb and P=gy (hydrostatic equilibrium)
yb = 1e8
gyb = G*M/R**2 * yb

from physics import EquationOfState
eos = EquationOfState(comp='He')

# In the derivation, we find an expession for rho(T,x)
def rho(T,x):
    return eos.mu*mp/kB * arad/3 * T**3 * x**(-1) * (1 - x + 4/(4+alpha)*(T/T0)**alpha)

def find_Tb(x):

    def innerBC_error(Tb):
        P = eos.pressure(rho(Tb,x), Tb, lam=1/3, R=0)
        return P-gyb
    
    return fsolve(innerBC_error, x0=1e9)


# We find the photospheric radius for any x by finding the root to the big equation

def find_rph(x):

    Tb = find_Tb(x)
    LHS = 1 - (1- 4/(4+alpha)/(1-x)) * hyp2f1(1, 1/alpha, 1+1/alpha, -4*(Tb/T0)**alpha/(1-x)/(4+alpha))

    return R / (1 - kB*Tb/(eos.mu*mp) * R/(G*M) * (4+alpha) * LHS)



# if x=1 exactly, we use P&A formula (otherwise there is division by zero)
def find_rph_x1():

    Tb = find_Tb(1)
    LHS = 1 + 1/(1-alpha) * (T0/Tb)**alpha

    return R / (1 - kB*Tb/(eos.mu*mp) * R/(6.6726e-8*M) * (4+alpha) * LHS)


# The analytical solution with the hypergeometric function was obtained with a symbolic solver
# (mathematica), let's verify that it's correct by plugging back into the ODE
def validate_equation(x):
    assert (x < 1)

    rph = find_rph(x)

    def find_r_for_T(T):
        def error(r):
            LHS = G*M/r * eos.mu*mp/kB/T * 1/(4+alpha) * (1-r/rph)
            RHS = 1 - (1- 4/(4+alpha)/(1-x)) * hyp2f1(1, 1/alpha, 1+1/alpha, -4*(T/T0)**alpha/(1-x)/(4+alpha))
            return LHS-RHS
        return fsolve(error, x0=R)

    Tb = find_Tb(x)
    T = np.ravel(np.linspace(Tb/1000,Tb,1000)) # ravel for flattened array
    r = []
    for Ti in T:
        r.append(find_r_for_T(Ti))
    r = np.ravel(r)

    # return r,T

    # Now plug back into ODE
    dr_dT = np.diff(r)/np.diff(T)
    r,T = r[1:], T[1:]
    term1 = (1+(T/T0)**alpha)/(1-x+4/(4+alpha)*(T/T0)**alpha) 
    term2 = 1/4 * eos.mu*mp/kB * G*M/r**2 * dr_dT
    err = 1+term2/term1 # this makes the error normalized
    print("The largest error obtained by plugging the analytical solution into the ODE is: %.3e"%np.max(np.abs(err)))

    # indeed by running this we see that the errors are small. the analytical solution is correct.