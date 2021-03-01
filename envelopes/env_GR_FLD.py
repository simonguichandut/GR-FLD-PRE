''' Main code to calculate expanded envelopes '''

import sys
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from collections import namedtuple
import numpy as np
import IO
import physics
from scipy.interpolate import InterpolatedUnivariateSpline as IUS 


# --------------------------------------- Constants and parameters --------------------------------------------

# Constants
kB = 1.380658e-16
arad = 7.5657e-15
c = 2.99792458e10
mp = 1.67e-24
kappa0 = 0.2
sigmarad = 0.25*arad*c

# Parameters
params = IO.load_params()

# Generate EOS class and methods
eos = physics.EOS(params['comp'])


# Mass-dependent parameters
M,RNS,yb = params['M'],params['R'],params['yb']
GM = 6.6726e-8*2e33*M
LEdd = 4*np.pi*c*GM/eos.kappa0
ZZ = (1-2*GM/(c**2*RNS*1e5))**(-1/2) # redshift
g = GM/(RNS*1e5)**2 * ZZ
Pb = g*yb

rg = 2*GM/c**2 # gravitationnal radius

# ----------------------------------------- General Relativity ------------------------------------------------

def Swz(r):  # Schwartzchild metric term
    return (1-2*GM/c**2/r)     

def grav(r): # local gravity
    return GM/r**2 * Swz(r)**(-1/2)

def Lcr(r,rho,T):
    return 4*np.pi*c*GM/eos.kappa(rho,T)*(1-rg/r)**(-1/2)

# ----------------------- Flux-limited diffusion -----------------------------
# Modified version of Pomraning (1983) FLD prescription.  
# See Guichandut & Cumming (2020)

def FLD_Lam(L,r,T,return_params=False):

    # L is the local luminosity. In envelopes, Linf is constant, L=Linf*(1-rs/r)**(-1)

    Flux = L/(4*np.pi*r**2)
    alpha = Flux/(c*arad*T**4)  # 0 opt thick, 1 opt thin

    if isinstance(L, (list,tuple,np.ndarray)): 
        if len(alpha[alpha>1])>0:
            raise Exception("Causality warning F>cE at %d locations"%len(alpha[alpha>1]))
    else:
        if alpha>1:
            # print('causality warning : F>cE')
            alpha=1-1e-9

    Lam = 1/12 * ( (2-3*alpha) + np.sqrt(-15*alpha**2 + 12*alpha + 4) )  # 1/3 thick , 0 thin
    R = alpha/Lam # 0 thick, 1/lam->inf thin

    if return_params:
        return Lam,alpha,R
    else:
        return Lam


# ----------------------------------------- Initial conditions ------------------------------------------------

def photosphere(Rphot,f0):

    ''' Finds photospheric density and temperature (eq 9a-b) for a given luminosity-like parameter f0 
        Also sets Linf, the luminosity seen by observers at infinity '''

    def Teff_eq(T): 
        return eos.kappa(0.,T) - (GM*c/(Rphot**2*sigmarad) * Swz(Rphot)**(-1/2) * (1-10**f0))/T**4  # Assuming 0 density for opacity, which is no problem at photosphere

    Tkeep1, Tkeep2 = 0.0, 0.0
    npoints = 10
    while Tkeep1 == 0 or Tkeep2 == 0:
        logT = np.linspace(6, 8, npoints)
        for T in 10**logT:
            foo = Teff_eq(T)
            if foo < 0.0:
                Tkeep1 = T
            if foo > 0.0 and Tkeep2 == 0:
                Tkeep2 = T
        npoints += 10

    T = brentq(Teff_eq, Tkeep1, Tkeep2, xtol=1e-10, maxiter=10000)
    rho = 2/3 * eos.mu*mp/(kB*T) * grav(Rphot)/eos.kappa(0.,T) * 10**f0          
    Linf = 4*np.pi*Rphot**2*sigmarad*T**4* Swz(Rphot)

    return rho,T,Linf

def get_f0(Rphot,Tphot,Linf): # the opposite of what photosphere does, i.e find f0 based on Linf and values at photosphere
    Lph = Linf*Swz(Rphot)**(-1)
    Lcrph = 4*np.pi*c*GM/eos.kappa(0.,Tphot)*(1-rg/Rphot)**(-1/2)
    return np.log10(1-Lph/Lcrph)



# --------------------------------------- Wind structure equations (v=0) ---------------------------------------

def Y(r): 
    return np.sqrt(Swz(r)) # v=0

def Tstar(L, T, r, rho):  
    return L/LEdd * eos.kappa(rho,T)/eos.kappa0 * GM/(4*r) *\
            3*rho/(arad*T**4) * Y(r)**(-1)

def A(T):
    return 1+1.5*eos.cs2(T)/c**2
# def A_e(rho,T):  
#     pe,_,[alpha1,_,f] = eos.electrons(rho,T)
#     return 1 + 1.5*eos.cs2_I(T)/c**2 + pe/(rho*c**2)*(f/(f-1) - alpha1)

# def B(T):
#     return eos.cs2(T)
# def B_e(rho,T): 
#     pe,_,[alpha1,alpha2,f] = eos.electrons(rho,T)
#     return eos.cs2_I(T) + pe/rho*(alpha1 + alpha2*f)

# def C(L,T,r,rho):
#     Lam,_,R = FLD_Lam(L,r,T,return_params=True)
#     b = eos.Beta(rho,T, lam=Lam, R=R)
#     return 1/Y(r) * L/LEdd * eos.kappa(rho,T)/eos.kappa0 * GM/r * \
#             (1 + b/(12*Lam*(1-b)))

# def C_e(L, T, r, rho):  

#     Lam,_,R = FLD_Lam(L,r,T,return_params=True)
#     _,_,[alpha1,_,_] = eos.electrons(rho,T)
#     bi,be = eos.Beta_I(rho, T, lam=Lam, R=R), eos.Beta_e(rho, T, lam=Lam, R=R)

#     return 1/Y(r) * L/LEdd * eos.kappa(rho,T)/eos.kappa0 * GM/r * \
#             (1 + (bi + alpha1*be)/(12*Lam*(1-bi-be)))


# -------------------------------------------- Calculate derivatives ---------------------------------------

def derivs(r,y,Linf):

    # Version with the wind equations (with A,B,C) and v=0
    rho,T = y[:2]
    
    # if rho<0:   
    #     rho=1e-10

    L = Linf*Swz(r)**(-1)
    Lam = FLD_Lam(L,r,T)
    # print('r=%.3e \t rho=%.3e \t T=%.3e \t lam=%.3f'%(r,rho,T,Lam))

    dlnT_dlnr = -Tstar(L, T, r, rho) / (3*Lam) - 1/Swz(r) * GM/c**2/r
    # dlnrho_dlnr = (-GM/Swz(r)/r * A_e(rho,T) + C_e(L,T,r,rho))/B_e(rho,T)
    dlnrho_dlnr = (-GM/Swz(r)/r * A(T) + C(L,T,r,rho))/B(T)

    dT_dr = T/r * dlnT_dlnr
    drho_dr = rho/r * dlnrho_dlnr

    return [drho_dr,dT_dr]

# --------------------------------------- Optically thin limit ---------------------------------------

def T_thin(Linf,r):
    return ( Linf*Swz(r)**(-1) / (4*np.pi*r**2*arad*c) )**0.25

def drho_thin(r,rho,Linf):

    L = Linf*Swz(r)**(-1)
    T = T_thin(Linf,r)

    Flux = L/(4*np.pi*r**2)
    alpha = Flux/(c*arad*T**4)  

    # Lam = eos.kappa(rho,T)*rho*r/2/Y(r)  # optically thin limit of lambda (it's not exactly zero)
    Lam = eos.kappa0*rho*r/2/Y(r)  # optically thin limit of lambda (it's not exactly zero)
    b = eos.Beta(rho,T, lam=Lam, R=alpha/Lam)
    C =  1/Y(r) * L/LEdd * eos.kappa(rho,T)/eos.kappa0 * GM/r * (1 + b/(12*Lam*(1-b)))

    dlnrho_dlnr = (-GM/Swz(r)/r * A(T) + C)/B(T)
    return rho/r * dlnrho_dlnr
    
# ---------------------------------------------- Integration -----------------------------------------------

def Shoot_in(rspan, rho0, T0, Linf):
    ''' Integrates in using r as the independent variable, until P=Pb or 
        we diverge in the wrong direction. We want to match the location of 
        p=Pb to the NS radius '''

    inic = [rho0, T0]

    def hit_innerPressure(r,Y,*args): return eos.pressure(Y[0],Y[1], lam=1/3, R=0)-Pb
    hit_innerPressure.terminal = True # stop integrating at this point

    def hit_lowdensity(r,Y,*args): 
        return Y[0]-rho0/100
    hit_lowdensity.terminal = True # stop integrating when hit small density (means about to crash)
    

    sol = solve_ivp(derivs, rspan, inic, args = (Linf,), method='Radau', 
            events = (hit_innerPressure, hit_lowdensity), dense_output=True, 
                    atol=1e-6, rtol=1e-6, max_step=1e5)
    return sol

def Error(r): # Evaluate normalized error on location of the base versus NS radius
    return (r[-1]-RNS*1e5)/(RNS*1e5)

def Shoot_out(rspan, rho0, T0, Linf, rtol=1e-6, max_step=1e5):
    ''' Integrates out using r as the independent variable, until the maximum radius,
        or until density goes to zero (or a minimum value).'''

    inic = [rho0, T0]

    def hit_zero_density(r,y,*args):
        return y[0]    
    hit_zero_density.terminal = True


    # Something that can happen that will make my algo fail is that in the outwards integration,
    # the bottom branch, the one that should diverge with the density going negative (and stop at zero)
    # will not diverge somehow and find a stable region where the temperature stays constant and the 
    # density goes up and down.
    # I want to trigger a stop when this happens, but the condition that I implement must not trigger
    # when we're on the upper branch, where density goes up until the step size becomes too small.
    # For example, a condition that triggers when drho/dr>0 is not good because it triggers in the other branch.
    
    # I can try and catch the up and down of rho, i.e trigger when drho/dr goes from + to -. That should signal 
    # than I'm on the bottom branch, because the top branch fails when drho/dr is + and just keeps increasing

    def hit_density_mountains(r,y,*args):
        rho,T = y
        dlnrho_dlnr = (-GM/Swz(r)/r * A(T) + C(Linf*Swz(r)**(-1),T,r,rho))/B(T)
        # if dlnrho_dlnr>0: print(dlnrho_dlnr)
        return dlnrho_dlnr
    hit_density_mountains.direction = -1
    hit_density_mountains.terminal = True

    # def hit_optically_thin(r,y,*args):
    #     T_thin = ( Linf*Swz(r)**(-1) / (4*np.pi*r**2*arad*c) )**0.25
    #     err = abs(y[1]-T_thin)/T_thin
    #     return err - rtol
    # hit_optically_thin.terminal = True

    sol = solve_ivp(derivs, rspan, inic, args=(Linf,), method='Radau', 
            events = (hit_zero_density,hit_density_mountains), dense_output=True, 
                    atol=1e-6, rtol=rtol, max_step=1e5)

    return sol

def Shoot_out_thin(rvec, rho0, Linf, rtol=1e-6, rho_min=1e-10):
    ''' Integrates out using r as the independent variable, using the optically thin 
        limit to calculate the temperature '''

    def hit_minimum_density(r,y,*args):
        return y[0]-rho_min      
    hit_minimum_density.terminal = True

    T = ( Linf*Swz(rvec)**(-1) / (4*np.pi*rvec**2*arad*c) )**0.25
    sol = solve_ivp(drho_thin, t_span=(rvec[0],rvec[-1]), y0=(rho0,), args=(Linf,), 
            events = (hit_minimum_density), method='Radau', dense_output=True, 
            rtol=rtol, max_step = max_step)
    rho = sol.sol(rvec)[0]

    print('density zero at r=%.4f km'%(sol.t[-1]/1e5))

    return rho,T


# ------------------------------------------------- Envelope ---------------------------------------------------

Env = namedtuple('Env',
            ['rphot','Linf','r','T','rho'])

def get_rhophf0rel(Rphotkm, rend=1e9, tol=1e-6, Verbose=0, f0min=-4.5, f0max=-3.7, npts=40, spacing='linear'):

    # find the value of rhoph that allow a solution to go to inf (to tol precision), for each value of f0
    if Verbose: print('\nRphot = %.2f km\n'%Rphotkm)

    if spacing=='linear':
        f0vals = np.linspace(f0min,f0max,npts)
    elif spacing=='log':
        f0vals = -1*np.logspace(np.log10(abs(f0min)),np.log10(abs(f0max)),npts)
    f0vals = np.round(f0vals,8) # 8 decimals

    rspan = (Rphotkm*1e5, rend)

    for f0 in f0vals:
        if Verbose: print('\nFinding rhoph for f0 = %.8f'%f0)

        # Start at the initial value given by the approximation for tau=2/3
        rhoph,Tph,Linf = photosphere(Rphotkm*1e5, f0)
        if Linf/LEdd>1: print('Warning: Linf/LEdd=%.5f'%(Linf/LEdd))

        a = rhoph
        # sola = solve_ivp(derivs, (Rphotkm*1e5,1e9), (a,Tph), args=(Linf,), 
        #                     events=(hit_zero_density), method='Radau', dense_output=True, rtol=tol)
        sola = Shoot_out(rspan=rspan, rho0=a, T0=Tph, Linf=Linf)

        if sola.status == 1: # hit zero density (intial rhoph is in the bottom branch)
            direction = +1
        else:
            direction = -1

        # Step either up or down in rhoph until we find other branch
        step = 0.5 # 50% update
        b = a
        while True:
            b *= 1 + direction*step  # either *1.5 or *0.5
            # solb = solve_ivp(derivs, (Rphotkm*1e5,1e9), (b,Tph), args=(Linf,), 
            #                 events=(hit_zero_density), method='Radau', dense_output=True, rtol=tol)
            solb = Shoot_out(rspan=rspan, rho0=b, T0=Tph, Linf=Linf)
            if solb.status != sola.status:
                break

        # Bissect to get two values of rhoph close to relative tolerance tol
        # print('\nBeginning Bissection')
        while abs(b-a)/a>tol:
            m = (a+b)/2
            # print('%.6e'%m)
            # solm = solve_ivp(derivs, (Rphotkm*1e5,1e9), (m,Tph), args=(Linf,), 
            #                 events=(hit_zero_density), method='Radau', dense_output=True, rtol=tol)
            solm = Shoot_out(rspan=rspan, rho0=m, T0=Tph, Linf=Linf)
            if solm.status == sola.status:
                a,sola = m,solm
            else:
                b,solb = m,solm

        # a the smaller one just to not get confused
        if a>b: (a,b) = (b,a) 

        if Verbose:
            print('\nInitial rhoph based on PA86 formula : \n%.6e'%rhoph)
            print('Final bounding values:\n%.6e\n%.6e'%(a,b))

        # Save one at a time
        IO.save_rhophf0rel(Rphotkm,[f0],[a],[b])

    IO.clean_rhophf0relfile(Rphotkm,warning=0)



def OuterBisection(Rphotkm, rho0, T0, Linf, rend=1e9, Verbose=False, tol=1e-4, return_stuff=False):

    if return_stuff: stuff=[]

    a = rho0
    sola = Shoot_out(rspan=(Rphotkm*1e5,rend), rho0=a, T0=T0, Linf=Linf)

    if sola.status == 0: 
        import matplotlib.pyplot as plt 
        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(6,8),sharex=True)
        ax1.set_ylabel('T')
        ax2.set_ylabel('rho')
        ax2.set_xlabel('r (km)')
        ax1.semilogy(sola.t/1e5,sola.y[1],'r-',lw=0.8)
        ax1.semilogy(sola.t/1e5,T_thin(Linf,sola.t),'r--',lw=0.8)
        ax2.semilogy(sola.t/1e5,sola.y[0],'b-',lw=0.8)
        fig,ax3=plt.subplots(1,1)
        ax3.plot(sola.t/1e5,np.gradient(sola.y[0]))
        plt.show()
        sys.exit('reached end of integration interval with root!')

    if sola.status == 1: # hit zero density (intial rhoph is in the bottom branch)
        direction = +1
    else:
        direction = -1

    # Step either up or down in rhoph until we find other branch
    step = 1e-6
    b = a
    i = 0
    while True:
        logb = np.log10(b) + direction*step
        b = 10**logb
        solb = Shoot_out(rspan=(Rphotkm*1e5,rend), rho0=b, T0=T0, Linf=Linf)
        if solb.status != sola.status:
            break

        i+=1
        if i==200:
            if Verbose: 
                print('Not able to find a solution that diverges in opposite \
                    direction after changing rhoph by 200 tolerances.  \
                    Problem in the rhoph-f0 interpolation')
            break

    # if sola was the high rhoph one, switch sola and solb (such that a is bottom branch)
    if direction == -1:
        (a,sola),(b,solb) = (b,solb),(a,sola)

    if Verbose:
        print('Two initial solutions. logrho values at photosphere:')
        print('sola:%.6f \t solb:%.6f'%(np.log10(a),np.log10(b)))

            
    if return_stuff:
        stuff.append([a,sola,b,solb])
        stuff.append([]) # will store the intermediate solutions into this list


    def check_convergence(sola,solb,rcheck):
        """ checks if two solutions are converged (similar rho, T) at some r """
        rhoa,Ta = sola.sol(rcheck)
        rhob,Tb = solb.sol(rcheck)
        if abs(rhoa-rhob)/rhoa < tol and abs(Ta-Tb)/Ta < tol:
            return True,rhoa,rhob,Ta,Tb
        else:
            return False,rhoa,rhob,Ta,Tb

    # Create Radius array on which we will save points. Want many points near the photoshere to aid convergence
    if Rphotkm > RNS+0.1:
        Rlin = np.linspace(Rphotkm*1e5, (Rphotkm+5)*1e5, 1000)
        Rlog = np.logspace(np.log10((Rphotkm+5)*1e5), np.log10(rend), 1000)
    elif Rphotkm <= RNS+0.1 and Rphotkm > RNS+0.01:
        Rlin = np.linspace(Rphotkm*1e5, (Rphotkm+2)*1e5, 2000)
        Rlog = np.logspace(np.log10((Rphotkm+1)*1e5), np.log10(rend), 1000)
    else:
        Rlin = np.linspace(Rphotkm*1e5, (Rphotkm+0.5)*1e5, 3000)
        Rlog = np.logspace(np.log10((Rphotkm+0.5)*1e5), np.log10(rend), 1000)

    R = np.concatenate((Rlin,Rlog[1:]))

    # Start by finding the first point of divergence
    # Npts = 5000
    # R = np.logspace(np.log10(Rphotkm*1e5),np.log10(rend),Npts)  


    for i,ri in enumerate(R):
        conv = check_convergence(sola,solb,ri)
        if conv[0] is False:
            i0=i            # i0 is index of first point of divergence
            if Verbose: print('First divergence at r = %.3e cm'%ri)
            break
        else:
            rhoa,rhob,Ta,Tb = conv[1:]


    # Construct initial arrays
    rho,T = sola.sol(R[:i0])
    def update_arrays(rho,T,sol,R,j0,jf):
        # Add new values from rho and T using ODE solution object. 
        # Radius points to add are from R[j0] to R[jf]
        rho_new,T_new = sol(R[j0:jf+1])  # +1 so R[jf] is included
        rho,T = np.concatenate((rho,rho_new)), np.concatenate((T,T_new))
        return rho,T

    # Begin bisection
    if Verbose:
        print('\nBeginning bisection')
        print('rconv (km) \t Step # \t Iter \t m \t dir')  
    a,b = 0,1
    step,count = 0,0
    i = i0
    rconv = R[i-1]  # converged at that radius
    rcheck = R[i]   # checking if converged at that radius
    do_bisect = True

    while rconv<rend:  

        if do_bisect: # Calculate a new solution from interpolated values
            
            m = (a+b)/2
            rhom,Tm = rhoa + m*(rhob-rhoa), Ta + m*(Tb-Ta)

            max_step = 1e3 if rconv < (RNS+2)*1e5 else 1e5

            solm = Shoot_out(rspan=(rconv,10*rend), rho0=rhom, T0=Tm, Linf=Linf, rtol=1e-10, max_step=max_step) 
            # go further than rmax to give it the chance to diverge either way

            if solm.status == 0: # Reached rend - done
                # rho,T = update_arrays(rho,T,solm.sol,R,i0,len(R))  
                #jf=len(R) so that it includes the last point of R
                raise Exception('reached end of integration interval  without \
                    reaching optically thin.. probably wrong')
                

            elif solm.status == 1:
                a,sola,sdir = m,solm,'^'

            elif solm.status == -1:
                b,solb,sdir = m,solm,'v'

        else:
            i += 1
            rconv,rcheck = R[i-1],R[i]

        conv = check_convergence(sola,solb,rcheck)
        if conv[0] is True:

            rhoa,rhob,Ta,Tb = conv[1:]
            a,b = 0,1 # reset bissection parameters
            step += 1 # update step counter
            count = 0 # reset iteration counter

            # Converged, so on next iteration just look further
            do_bisect = False 

            # Store solutions for demo plot    
            if return_stuff:
                stuff[-1].extend((sola,solb))
        
        else:
            count+=1
            do_bisect = True

            # Before computing new solution, add converged results to array 
            # (but only if we made at least 1 step progress)
            if i-1>i0:
                rho,T = update_arrays(rho,T,solm.sol,R,i0,i-1)  # i-1 is where we converged last
                i0=i # next time we append

            # Check if we have reached the optically thin limit of temperature profile
            # but only if we've been stuck for a while and can't make progress with the normal equations
            if count == 100:
                rx,rhox,Tx = R[len(rho)-1], rho[-1], T[-1]
                if abs(Tx-T_thin(Linf,rx))/T_thin(Linf,rx) < 1e-3:
                    if Verbose: print('Reached optically thin limit at r=%.4f km'%(rx/1e5))
                
                    # rho3,T3 = Shoot_out_thin(R2[len(rho2)-1:], rho0, Linf)
                    # stuff.append([R2[len(rho2)-1:],rho3,T3])
                    # rho2,T2 = np.append(rho2, rho3[1:]) , np.append(T2, T3[1:])

                    # Append optically thin limit: T \propto r^-1/2, rho\approx 0
                    # Rthin = R[len(rho)-1:]
                    Rthin = np.logspace(np.log10(rx), np.log10(rend), 50) # don't need many points if it's analytical
                    rhothin = 1e-20 * np.ones(len(Rthin))
                    Tthin = T_thin(Linf,Rthin)

                    if return_stuff: stuff.append([Rthin,rhothin,Tthin])
                    R = np.concatenate((R[:len(rho)] , Rthin[1:]))
                    rho,T = np.append(rho, rhothin[1:]) , np.append(T, Tthin[1:])
                    assert(len(R)==len(rho))
                    break
            

        # Exit after stuck at 200 stuff for debugging (with 'stuff' object)
        if return_stuff and count==500:
            return R[:len(rho)],rho,T,stuff

        # Exit if stuck at one step
        nitermax=1000
        if count==nitermax:
            sys.exit("Could not integrate out to rend! Exiting after being \
                        stuck at the same step for %d iterations"%nitermax)

        # End of step
        if Verbose: print('%.4e \t %d \t\t %d \t\t %.10f \t\t %s'%(rconv,step,count,m,sdir))

    if return_stuff:
        return R,rho,T,stuff
    else:
        return R,rho,T



            
def MakeEnvelope(Rphotkm, rend=1e9, Verbose=False, tol=1e-4, return_stuff=False): 

    global Linf             # that way Linf does not have to always be a function input parameter
    Rphot = Rphotkm*1e5
    rspan = (Rphot , 1.01*rg)                       # rspan is the integration range

    stuff = []

    # Load relation between f0 and logrhoph
    rel = IO.load_rhophf0rel(Rphotkm)
    if rel[0] is False:
        if Verbose: print('First need to find relation between f0 and rhoph that allows integration to infinity')
        
        if Rphotkm>=RNS+1:
            get_rhophf0rel(Rphotkm, Verbose=Verbose)
        elif Rphotkm<RNS+0.02:
            get_rhophf0rel(Rphotkm, f0max=-1e-4, f0min=-1, npts=100, Verbose=Verbose, spacing='log')
        else:
            get_rhophf0rel(Rphotkm, f0max=-1e-3, npts=100, Verbose=Verbose)

        rel = IO.load_rhophf0rel(Rphotkm)

    if Verbose: print('\nLoaded rhoph-f0 relation from file')
    _,f0vals,rhophA,_ = rel

    rel_spline = IUS(f0vals[::-1],rhophA[::-1])

    
    # First pass to find border values of f, and their solutions sola (gives r(y8)<RNS) and solb (r(y8)>RNS)
    for i,f0 in enumerate(f0vals):
        _,T_phot,Linf = photosphere(Rphot,f0)
        rho_phot = rhophA[i]
        solb = Shoot_in(rspan,rho_phot,T_phot,Linf) 
        Eb = Error(solb.t)

        if i==0 and (Eb<0 or len(solb.t_events[1]) == 1):
            # raise Exception('Highest f0 value leads to rb<RNS, rerun get_rhophf0rel() with custom bounds')
            print('Highest f0 value leads to rb<RNS, rerun get_rhophf0rel() with custom bounds')
            get_rhophf0rel(Rphotkm,f0min=f0vals[0],f0max=min(f0vals[0]+1,-1e-3), Verbose=Verbose, npts=15)
            # just restart the whole function at this point
            # return MakeEnvelope(Rphotkm, rend=rend, Verbose=Verbose, tol=tol, return_stuff=return_stuff)
            # actually doesn't work
            raise Exception('Call Again')

        if Eb<0 or len(solb.t_events[1]) == 1: 
            # Either we crossed RNS or density went down and we stopped integration, so we didn't end up 
            # landing on rb<RNS. We will still keep this solution
            a,b = f0vals[i],f0vals[i-1]
            Ea,sola = Eb,solb
            solb,Eb = solprev,Eprev
            break
        solprev,Eprev = solb,Eb

    if Verbose: print('\nBorder values of f on first pass: %.6f \t %.6f\n'%(a,b))

    # In FLD this first pass is more tricky because the density is going to zero in one of the branches.
    # We need to do a second pass to find very closer border values for f
    while abs((b-a)/b)>1e-6:
        a = b - abs(a-b)/2
        _,T_phot,Linf = photosphere(Rphot,a)
        rho_phot = rel_spline(a)

        sola=Shoot_in(rspan,rho_phot,T_phot,Linf)
        if len(sola.t_events[0])==1:  # hit inner pressure

                if Error(sola.t)<0:
                    Ea,Eb = Error(sola.t),Error(solb.t)
                    break
                else:
                    b,solb=a,sola
                    a=b-0.01
                    Ea,Eb = -1,Error(solb.t)  # -1 to have a negative value, the integration cannot get passed R
    
    assert((Ea<0) and (Eb>0))

    if Verbose:
        print('\nNarrowed down initial values for f at photosphere to:')
        if Ea==-1:
            print('fa=%.6f -> crashed to low density at r=%.3f km'%(a,sola.t_events[1][0]/1e5))
        else:
            print('fa=%.6f -> hit inner pressure at at r=%.3f km'%(a,sola.t_events[0][0]/1e5))
        print('fb=%.6f -> hit inner pressure at r=%.3f km\n\n'%(b,solb.t_events[0][0]/1e5))

    if return_stuff:
        stuff.append([a,sola,b,solb])
        stuff.append([]) # will store the intermediate solutions into this list

        
    def check_convergence(sola,solb,rcheck_prev):  
        ''' Checks if two solutions have similar parameters rho,T (1 part in tol^-1), some small integration distance in. 
            If converged, returns the interpolated value of rho,T at that point                            '''
        d = Rphot/100/(count+1) # 1% of photosphere at a time, reduce by count number of current iteration
        rcheck = rcheck_prev - d
        rhoa,Ta = sola.sol(rcheck)
        rhob,Tb = solb.sol(rcheck)
        
        if abs(rhoa-rhob)/rhoa < tol and abs(Ta-Tb)/Ta < tol:
            return True,rcheck,rhoa,rhob,Ta,Tb
        else:
            return False,
        

    # Begin bisection 
    Em=100
    count_iter,count = 0,0
    rhoa,rhob,Ta,Tb = [0 for i in range(4)]
    f0 = b

    r,rho,T = [np.array([]) for i in range(3)]       
 
    while abs(Em)>tol:   # we can stop when the final radius is the neutron star radius close to one part in 10^5
        
        # middle point.  In the first integration, a&b are the f values.  In the rest, a&b are between 0 and 1. For interpolation
        m = (a+b)/2
    
        if count_iter == 0:  # in the first iteration, a&b represent f
            _,T_phot,Linf = photosphere(Rphot,m)
            rho_phot = rel_spline(m)
            solm = Shoot_in(rspan,rho_phot,T_phot, Linf)
            rcheck = Rphot
        else:      # in the other iterations, a&b represent the linear space in [rhoa,rhob] and [Ta,Tb]
            rhom,Tm = rhoa + m*(rhob-rhoa) , Ta + m*(Tb-Ta)
            solm = Shoot_in(rspan,rhom,Tm,Linf)

        if count_iter == 1:
            if Verbose: 
                print('We begin the bissection with values for the photoshere')
                print('f = %.3e \t Linf = %.3e \t Tph = %.3e \t rhoph = %.3e'%(f0,Linf,T_phot,rho_phot))
            L = Linf*Swz(Rphot)**(-1)
            F = L/(4*np.pi*Rphot**2)
            alpha = F/(c*arad*T_phot**4)
            Lam = FLD_Lam(L,Rphot,T_phot)
            if Verbose:
                print('alpha = %.3f \t lambda = %.3f'%(alpha,Lam))
                print('L/4pir^2sigT^4 = %.3f'%(F/sigmarad/T_phot**4))
                print('\nRadius (km) \t Step # \t Iter count \t RNS error')   
        
            
        # Bisection : check which side the new solution lands on and update either a or b
        if len(solm.t_events[0])==1:  # hit inner pressure
            Em = Error(solm.t)
        else:
            Em = -1  # just need to have a negative value

        if Ea*Em>0:
            a,sola = m,solm
        else:
            b,solb = m,solm


        conv = check_convergence(sola,solb,rcheck)
        # When the two solutions are converging on rho and T, move the starting point inwards and reset a & b
        if conv[0] is True:
            rcheck,rhoa,rhob,Ta,Tb = conv[1:]
            rspan = (rcheck,1.01*rg)
            a,b = 0,1  
            count_iter+=1  # update step counter
            count = 0      # reset iteration counter
            
            r, rho, T = np.append(r,rcheck), np.append(rho,(rhoa+rhob)/2), np.append(T,(Ta+Tb)/2)
        

            if return_stuff:
                stuff[1].extend((sola,solb))


        # End of step 
        if Verbose: print('%.5f \t %d \t\t %d \t\t %.6e'%(rcheck/1e5,count_iter,count+1,Em))
        count+=1

        # Exit if stuck at a step
        nitermax=1000
        if count==nitermax:
            sys.exit("Could not arrive at the neutron star radius! Exiting after being stuck at the same step for %d iterations"%nitermax)

    # Reached precision criteria for error on radius
    if Verbose: print('Reached surface at r=%.5f km!\n'%(solm.t[-1]/1e5))

    # Fill out arrays    
    r,rho,T  = np.insert(r,0,Rphot), np.insert(rho,0,rho_phot), np.insert(T,0,T_phot)
    ind = solm.t<r[-1]
    r,rho,T  = np.append(r,solm.t[ind]), np.append(rho,solm.y[0][ind]), np.append(T,solm.y[1][ind])    
    r,rho,T = np.flip(r),np.flip(rho),np.flip(T)

    # Then make a solution to rend by bisection
    if return_stuff:
        r2,rho2,T2,stuff2 = OuterBisection(Rphotkm, rho0=rho[-1], T0=T[-1], Linf=Linf, rend=rend, Verbose=Verbose, return_stuff=True)
        r,rho,T = np.append(r,r2[1:]), np.append(rho,rho2[1:]), np.append(T,T2[1:])
        stuff.extend(stuff2)
        return Env(Rphot,Linf,r,T,rho),stuff
    
    else:
        r2,rho2,T2 = OuterBisection(Rphotkm, rho0=rho[-1], T0=T[-1], Linf=Linf, rend=rend, Verbose=Verbose, return_stuff=False)
        r,rho,T = np.append(r,r2[1:]), np.append(rho,rho2[1:]), np.append(T, T2[1:])
        return Env(Rphot,Linf,r,T,rho)

