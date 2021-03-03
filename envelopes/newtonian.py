import sys
sys.path.append(".")

## Paczynski and Anderson gave a simple formula for static envelopes when L\approx LEdd
## Following their derivations but keeping x=L/LEdd a free parameter, the result depends on the hypergeometric function (see our Appendix D)

from scipy.special import hyp2f1
from scipy.optimize import fsolve

# Neutron star parameters
M = 1.4*2e33 # g
R = 12e5 # cm

# Constants
kB = 1.380658e-16
arad = 7.5657e-15
c = 2.99792458e10
mp = 1.67e-24

# Opacity formula : kappa=kappa0 / (1 + (T/T0)^alpha)
T0 = 4.5e8 # K
alpha = 0.86 

# To solve, we need to know the temperature at the base (rb=R). We use the BC that y=yb and P=gy (hydrostatic equilibrium)
yb = 1e8
gyb = 6.6726e-8*M/R**2 * yb

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

    return R / (1 - kB*Tb/(eos.mu*mp) * R/(6.6726e-8*M) * (4+alpha) * LHS)



# if x=1 exactly, we use P&A formula (otherwise there is division by zero)
def find_rph_x1():

    Tb = find_Tb(1)
    LHS = 1 + 1/(1-alpha) * (T0/Tb)**alpha

    return R / (1 - kB*Tb/(eos.mu*mp) * R/(6.6726e-8*M) * (4+alpha) * LHS)
