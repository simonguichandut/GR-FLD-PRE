## Physics for this problem : equation of state (EOS)

from numpy import sqrt,pi,ndarray

# Constants
kB = 1.380658e-16
arad = 7.5657e-15
c = 2.99792458e10
mp = 1.67e-24

# some functions have a **kwargs argument which is there so that dummy arguments can be 
# put in the function call without adding errors. In one EOS, Beta could have 2 args, and in the
# other it could have 4 args. With kwargs we can always call it with 4 args and the last 2 arguments
# might just be dummy arguments. In that case these last two arguments must be called with key=arg (kwarg)


class EquationOfState:

    # Ideal gas equation of state with variable radiation pressure, Pr=faT^4, where f=1/3 in optically thick regions, f=1 in optically thin.


    def __init__(self, comp):

        self.comp = comp

        # X,Y,Z : H, He, metals fraction (X+Y+Z=1)
        # Zno : atomic number
    
        # Homogeneous
        if self.comp in ('He','H','Ni'):

            if self.comp == 'He':
                self.X = 0
                self.Y = 1
                self.Z = 0
                self.Zno = 2
                self.mu_I = 4

            elif self.comp == 'H' :
                self.X = 1
                self.Y = 0
                self.Z = 0
                self.Zno = 1
                self.mu_I = 1

            elif self.comp == 'Ni':
                self.X = 0
                self.Y = 0
                self.Z = 1
                self.Zno = 28
                self.mu_I = 56

            self.mu_e = 2/(1+ self.X)                                   # http://faculty.fiu.edu/~vanhamme/ast3213/mu.pdf
            self.mu = 1/(1/self.mu_I + 1/self.mu_e)

        # Heterogeneous
        else:
            if self.comp.lower() == 'solar':
                self.X, self.Y, self.Z = 0.7, 0.28, 0.02            # http://www.astro.wisc.edu/~townsend/resource/teaching/astro-310-F08/21-eos.pdf
            else:
                # Expecting "X,Y,Z"
                self.X, self.Y, self.Z = eval(comp)

            self.mu = 1/(2*self.X + 3*self.Y/4 + self.Z/2)        # http://www.astro.wisc.edu/~townsend/resource/teaching/astro-310-F08/21-eos.pdf
            self.mu_e = 2/(1+ self.X)
            self.mu_I = 1/(1/self.mu - 1/self.mu_e)

        self.kappa0 = 0.2 * (1+self.X)                              # https://www.astro.princeton.edu/~gk/A403/opac.pdf


    # Opacitioes
    def kff(self,rho,T):
        if self.comp in ('He','H','Ni'):
            return 1e23*self.Zno**2/(self.mu_e*self.mu_I)*rho*T**(-7/2)
        else:
            return 3.68e22 * (1-self.Z)*(1+self.X)*rho*T**(-7/2)

    def kes(self,T):
        return self.kappa0/(1.0+(T/4.5e8)**0.86) 

    def kappa(self,rho,T):
        return self.kes(T)    
        # return self.kes(rho,T) + self.kff(rho,T)

    # Ideal gas sound speed c_s^2
    def cs2(self,T): 
        return kB*T/(self.mu*mp)

    # Radiation pressure
    def rad_pressure(self, T, lam, R): 
        # lambda and R are the flux-limited diffusion parameters (Levermore & Pomraning)
        return (lam + (lam*R)**2)*arad*T**4
    

    # P and E for ideal gas + radiation
    def pressure(self, rho, T, lam, R):  
        return rho*self.cs2(T) + self.rad_pressure(T,lam,R)

    def internal_energy(self, rho, T):  
        return 1.5*self.cs2(T)*rho + arad*T**4 

    # Rest energy + enthalpy
    def H(self, rho, T, lam, R): 
        return c**2 + (self.internal_energy(rho, T) + self.pressure(rho, T, lam, R))/rho

    # Pressure ratio
    def Beta(self, rho, T, lam, R):  # pressure ratio 
        Pg = rho*self.cs2(T)
        Pr = self.rad_pressure(T,lam,R)
        return Pg/(Pg+Pr)



class GeneralRelativity():

    def __init__(self, M):

        self.M = M
        self.GM = 6.6726e-8*2e33*M
        self.rg = 2*self.GM/c**2

    def Swz(self, r):                # Schwartzchild metric term
        return (1-self.rg/r)

    def grav(self,r):                    # local gravity
        return self.GM/r**2 * self.Swz(r)**(-1/2)

    def gamma(self, v):              # Lorentz Factor
        return 1/sqrt(1-v**2/c**2)

    def Y(self, r, v):                # Energy parameter
        return sqrt(self.Swz(r))*self.gamma(v)

    def Lcrit(self, r, rho, T, eos):        # Local critical luminosity
        return 4*pi*c*self.GM/eos.kappa(rho,T) * self.Swz(r)**(-1/2)

    def Lcomoving(self, Lstar, r, v):  # Comoving luminosity from Lstar (Lstar = Linf if v=0)
        return Lstar/(1+v**2/c**2)/self.Y(r, v)**2



class FluxLimitedDiffusion():

    ''' Modified flux-limited diffusion of Levermore & Pomraning (1981) for GR '''

    def __init__(self, GR):
        self.GR = GR # GeneralRelativity class gives Lcomoving

    def x(self, Lstar, r, T, v):

        L = self.GR.Lcomoving(Lstar,r,v)
        Flux = L/(4*pi*r**2)
        x = Flux/(c*arad*T**4)  # 0 opt thick, 1 opt thin

        return x


    def Lambda(self, Lstar, r, T, v, return_params=False):

        x = self.x(Lstar,r,T,v)

        if isinstance(Lstar, (list,tuple,ndarray)): 
            if len(x[x>1])>0:
                raise Exception("Causality warning F>cE at %d locations"%len(x[x>1]))
        else:
            if x>1:
                # print('causality warning : F>cE')
                x=1-1e-9

        Lam = 1/12 * ( (2-3*x) + sqrt(-15*x**2 + 12*x + 4) )  # 1/3 thick , 0 thin
        R = x/Lam # 0 thick, 1/lam->inf thin

        if return_params:
            return Lam,x,R
        else:
            return Lam


class GradientParameters():

    def __init__(self,M,eos,GR,FLD):

        self.eos = eos        
        self.GM = 6.6726e-8*2e33*M
        self.LEdd = 4*pi*c*self.GM / self.eos.kappa0

        self.GR = GR
        self.FLD = FLD

    def A(self, T): 
        return 1 + 1.5*self.eos.cs2(T)/c**2

    def B(self, T):
        return self.eos.cs2(T)

    def C(self, Lstar, T, r, rho, v):  

        lam,_,R = self.FLD.Lambda(Lstar,r,T,v,return_params=True)
        L = self.GR.Lcomoving(Lstar,r,v)
        b = self.eos.Beta(rho,T, lam=lam, R=R)

        return 1/self.GR.Y(r,v) * L/self.LEdd * self.eos.kappa(rho,T)/self.eos.kappa0 * \
                self.GM/r * (1 + b/(12*lam*b))


    def Tstar(self, Lstar, T, r, rho, v): 

        lam,_,_ = self.FLD.Lambda(Lstar,r,T,v,return_params=True)
        L = self.GR.Lcomoving(Lstar,r,v)

        return 1/(self.GR.Y(r,v)*lam) * L/self.LEdd * self.eos.kappa(rho,T)/self.eos.kappa0 \
                 * self.GM/(4*r) * rho/(arad*T**4)
