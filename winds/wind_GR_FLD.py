''' 
Main code to calculate winds 
Version with flux-limited diffusion : transitions to optically thin
'''

import sys
sys.path.append(".")

import numpy as np
from scipy.optimize import brentq,fsolve,toms748,root_scalar
from scipy.integrate import solve_ivp,quad
from collections import namedtuple
from scipy.interpolate import InterpolatedUnivariateSpline as IUS 
import IO
import physics

# ----------------------- Constants and parameters ---------------------------

# Constants
arad = 7.5657e-15
c = 2.99792458e10
sigmarad = 0.25*arad*c

# Parameters
params = IO.load_params()

# EOS class and methods
eos = physics.EquationOfState(params['comp'])

# General relativity functions
gr = physics.GeneralRelativity(params['M'])

# Flux-limited diffusion
fld = physics.FluxLimitedDiffusion(gr)

# Gradient equation parameters
F = physics.GradientParameters(params['M'],eos,gr,fld)

LEdd = F.LEdd

# Base boundary condition
M,RNS,yb = params['M'],params['R'],params['yb']
GM = 6.6726e-8*2e33*M
Pb = gr.grav(RNS*1e5)*yb
rhomax = 1e7


# ----------------------------- Wind specific ---------------------------------

def solve_energy(r,v,T):

    if params['Prad'] == 'simple':
        # If Prad=aT^4 everywhere, the Edot equation is straightforward to solve for Lstar
        rho = Mdot/(4*np.pi*r**2*gr.Y(r, v)*v)
        return Edot - Mdot*gr.Y(r,v)*eos.H(rho,T,lam=1/3,R=0)


    elif params['Prad'] == 'exact':

        # If Prad = (lambda+lambda^2*R^2)aT^4, the Edot equation is implicit

        # If fed arrays, need to go one by one
        if isinstance(r, (list, tuple, np.ndarray)):
            Lstar = []
            for i in range(len(r)):
                Lstar.append(solve_energy(r[i],v[i],T[i]))
            return Lstar    

        rho = Mdot/(4*np.pi*r**2*gr.Y(r, v)*v)

        Lstar1 = Edot - Mdot*eos.H(rho, T, lam=1/3, R=0)*gr.Y(r, v)   # optically thick
        Lstar2 = Edot - Mdot*eos.H(rho, T, lam=1e-10, R=1e10)*gr.Y(r, v)  # optically thin
        def energy_error(Lstar):
            Lam,R = fld.Lambda(Lstar,r,T,v)
            return Edot - Mdot*eos.H(rho, T, Lam, R)*gr.Y(r, v) - Lstar

        Lstar = brentq(energy_error, Lstar1, Lstar2, xtol=1e-6, rtol=1e-8) #brentq is fastest
        x = fld.x(Lstar,r,v,T)
        # if x-1>1e-3:
        #     print('causality warning : F>cE (x-1=%.1e) (after solve_energy!?)'%(x-1))
        return Lstar

def uphi(phi, T, subsonic):
    ''' phi should never drop below 2, but numerically
    it is sometimes just < 2, so in that case return Mach number 1 (divided by 
    sqrt(A)). This is using the GR version of the Joss & Melia change of var
    phi in sonic units, where the difference between the velocity and sound 
    speed at the critical point (vs=sqrt(B)/sqrt(A)=0.999cs) is taken into 
    account : phi = sqrt(A)*mach + 1/sqrt(A)/mach '''

    if phi < 2.0:   
        u = 1.0*np.sqrt(F.B(T)/np.sqrt(F.A(T)))
    else:
        if subsonic:
            u = 0.5*phi*np.sqrt(F.B(T))*(1.0-np.sqrt(1.0-(2.0/phi)**2))/np.sqrt(F.A(T))
        else:
            u = 0.5*phi*np.sqrt(F.B(T))*(1.0+np.sqrt(1.0-(2.0/phi)**2))/np.sqrt(F.A(T))
    return u

# ----------------------------- Sonic point ---------------------------------

def numerator(r, T, v):  
    
    rho = Mdot/(4*np.pi*r**2*gr.Y(r, v)*v)     
    Lstar = solve_energy(r,v,T)

    return gr.gamma(v)**(-2) *\
           (GM/r/gr.Swz(r) * (F.A(T)-F.B(T)/c**2) - F.C(Lstar, T, r, rho, v) - 2*F.B(T))

def rSonic(Ts):

    rkeep1, rkeep2 = 0.0, 0.0
    npoints = 50
    vs = np.sqrt(eos.cs2(Ts)/F.A(Ts))

    # First check if sonic point would be below 2 x gravitational radius
    # NS radius should never be under 2rg anyway
    rg = 2*GM/c**2
    if numerator(2*rg, Ts, vs) > 0.0:
        raise Exception("Error: sonic point is below gravitational radius")
    
    # Check if it's below RNS
    if numerator(RNS*1e5, Ts, vs) > 0.0:
        if verbose: print("Warning : sonic point below RNS")

    while rkeep1 == 0 or rkeep2 == 0:
        logr = np.linspace(np.log10(2*rg), 9, npoints)
        for r in 10**logr:
            try:
                foo = numerator(r, Ts, vs)
            except Exception:
                print('Broke causality (F>cE) when trying sonic pt at r=%.3e'%r)
                pass
            else:
                if foo < 0.0:
                    rkeep1 = r
                if foo > 0.0 and rkeep2 == 0:
                    rkeep2 = r
        
        npoints += 10
    # print('sonic: rkeep1 = %.3e \t rkeep2 = %.3e'%(rkeep1,rkeep2))
    # global rs

    rs = brentq(numerator, rkeep1, rkeep2, args=(Ts, vs), maxiter=100000)
    return rs

# -------------------- Calculate vars and derivatives -------------------------

def calculateVars_phi(r, T, phi, subsonic=False, return_all=False):  

    # At the end phi will be given as an array
    if isinstance(phi, (list, tuple, np.ndarray)):
        u = []
        for i in range(len(phi)):
            u.append(uphi(phi[i], T[i], subsonic))
        r, T, u = np.array(r), np.array(T), np.array(u)
    else:
        u = uphi(phi, T, subsonic)

    rho = Mdot/(4*np.pi*r**2*u*gr.Y(r, u))
    Lstar = solve_energy(r,u,T)   

    if not return_all:
        return u, rho, phi, Lstar
    else:
        lam,R = fld.Lambda(Lstar,r,T,u)
        P = eos.pressure(rho, T, lam=lam, R=R)
        L = gr.Lcomoving(Lstar,r,u)
        cs = np.sqrt(eos.cs2(T))
        return u, rho, phi, Lstar, L, P, cs, lam


def calculateVars_rho(r, T, rho, return_all=False): 

    # At the end rho will be given as an array
    if isinstance(rho, (list, tuple, np.ndarray)):
        r, T, rho = np.array(r), np.array(T), np.array(rho)  

    u = Mdot/np.sqrt((4*np.pi*r**2*rho)**2*gr.Swz(r) + (Mdot/c)**2)
    Lstar = solve_energy(r,u,T)   

    if not return_all:
        return u, rho, Lstar
    else:

        lam,R = fld.Lambda(Lstar,r,T,u)

        mach = u/np.sqrt(F.B(T))
        phi = np.sqrt(F.A(T))*mach + 1/(np.sqrt(F.A(T))*mach)
        P = eos.pressure(rho, T, lam=lam, R=R)    

        L = gr.Lcomoving(Lstar,r,u)
        cs = np.sqrt(eos.cs2(T))
    
        return u, rho, phi, Lstar, L, P, cs, lam
   

def dr(r, y, subsonic):
    ''' Derivatives of T and phi by r '''

    T, phi = y[:2]
    u, rho, phi, Lstar = calculateVars_phi(r, T, phi=phi, subsonic=subsonic)

    x = fld.x(Lstar,r,T,u)
    if x-1>1e-4:
        print('causality warning : F>cE (x-1=%.1e) (during integration)'%(x-1))

    Lam,_ = fld.Lambda(Lstar,r,T,u)
    dlnT_dlnr = -F.Tstar(Lstar, T, r, rho, u) - GM/c**2/r/gr.Swz(r)
    # remove small dv_dr term which has numerical problems near sonic point
    dT_dr = T/r * dlnT_dlnr

    mach = u/np.sqrt(F.B(T))
    dphi_dr = (F.A(T)*mach**2 - 1) *\
        ( 3*F.B(T) - 2*F.A(T)*c**2) / (4*mach*F.A(T)**(3/2)*c**2*r) * dlnT_dlnr \
        - numerator(r, T, u) / (u*r*np.sqrt(F.A(T)*F.B(T)) )
    
    return [dT_dr, dphi_dr]

def dr_wrapper_supersonic(r,y): return dr(r,y,subsonic=False)
def dr_wrapper_subsonic(r,y):   return dr(r,y,subsonic=True)

def drho(rho, y):
    ''' Derivatives of T and r by rho '''

    T, r = y[:2]
    u, rho, Lstar = calculateVars_rho(r, T, rho = rho)

    # Not using phi
    dlnT_dlnr = -F.Tstar(Lstar, T, r, rho, u) - 1/gr.Swz(r) * GM/c**2/r
    dT_dr = T/r * dlnT_dlnr

    dlnr_dlnrho = (F.B(T) - F.A(T)*u**2) / \
            ((2*u**2 - (GM/(r*gr.Y(r, u)**2))) * F.A(T) + F.C(Lstar, T, r, rho, u)) 

    dr_drho = r/rho * dlnr_dlnrho
    dT_drho = dT_dr * dr_drho

    return [dT_drho, dr_drho]

# ------------------------------ Integration ---------------------------------

def outerIntegration(r0, T0, phi0, rmax=1e10):
    ''' Integrates out from r0 to rmax tracking T and phi '''

    if verbose:
        print('**** Running outerIntegration ****')

    # Stopping events
    def hit_mach1(r,y): 
        if r>5*rs:
            return y[1]-2  # mach 1 
        else: 
            return 1
    hit_mach1.terminal = True # stop integrating at this point
   
    # def hit_1e8(r,y):
    #     return uphi(y[1],y[0],subsonic=False)-1e8
    # hit_1e8.direction = -1
    # hit_1e8.terminal = True
    
    def dv_dr_zero(r,y):
        if r>5*rs:
            return numerator(r,y[0],uphi(y[1],y[0],subsonic=False))
        else:
            return -1
    dv_dr_zero.direction = +1  
    # triggers when going from negative to positive 
    # (numerator is negative in supersonic region)
    dv_dr_zero.terminal = True
        
    # Go
    inic = [T0,phi0]
    sol = solve_ivp(dr_wrapper_supersonic, (r0,rmax), inic, method='Radau', 
                    events=(dv_dr_zero,hit_mach1), dense_output=True, 
                    atol=1e-6, rtol=1e-10, max_step=1e6)
    
    if verbose: 
        print('FLD outer integration : ',sol.message, ('rmax = %.3e'%sol.t[-1]))
    return sol


def innerIntegration_r(rs, Ts):
    ''' Integrates in from the sonic point to 95% of the sonic point, 
        using r as the independent variable '''
    
    if verbose:
        print('**** Running innerIntegration R ****')

    inic = [Ts, 2.0]

    # sol = solve_ivp(dr_wrapper_subsonic, (rs,0.95*rs), inic, method='RK45',
    #                 atol=1e-6, rtol=1e-6, dense_output=True)   

    sol = solve_ivp(dr_wrapper_subsonic, (rs,0.95*rs), inic, method='Radau',
                atol=1e-6, rtol=1e-10, dense_output=True, first_step=1e3, max_step=1e4)   

    if verbose: print(sol.message)

    return sol


def innerIntegration_rho(rho95, T95, returnResult=False):
    ''' Integrates in from 0.95rs, using rho as the independent variable, 
        until rho=rhomax. Want to match the location of p=p_inner to the RNS '''

    if verbose:
        print('**** Running innerIntegration RHO ****')

    inic = [T95, 0.95*rs]

    flag_u0 = 0

    def hit_Pinner(rho,y):              
        # Inner boundary condition
        # Will be optically thick there so no worries with FLD
        T = y[0]
        P = eos.pressure(rho,T,lam=1/3,R=0)
        return P-Pb
    hit_Pinner.terminal = True

    def hit_zerospeed(rho,y):           # Don't want u to change sign
        r = y[1]
        u = Mdot/np.sqrt((4*np.pi*r**2*rho)**2*gr.Swz(r) + (Mdot/c)**2)
        return u
    hit_zerospeed.terminal = True        

    # Issue with solve_ivp in scipy 1.3.0 (fixed in yet to be released 1.4.0) 
    # https://github.com/scipy/scipy/pull/10802. # Will have a typeError when 
    # reaching NaNs, and won't return the result properly.
    
    try:
        sol = solve_ivp(drho, (rho95,rhomax), inic, method='Radau',
                        events = (hit_Pinner,hit_zerospeed),  dense_output=True,
                        atol=1e-6, rtol=1e-6,)    
    except:
        if verbose: 
            print('Surface pressure never reached (NaNs before reaching p_inner)')
        return +200


    if verbose: print(sol.message)

    if sol.status == 1 :         # A termination event occured

        if len(sol.t_events[0]) == 1:  # The correct termination event 
            
            rbase = sol.y[1][-1]

            if returnResult:
                return sol
            else:
                return (rbase/1e5 - RNS)/RNS    # Boundary error #2

        else:
            flag_u0 = 1
            p = hit_Pinner(sol.t[-1],sol.y[-1]) + Pb
            col = p/g
            if verbose: print('Zero velocity before pressure condition reached.\
                                Last pressure : %.3e (y = %.3e)\n'%(p,col))
        
    else:
        if verbose: 
            print('Pressure condition nor zero velocity reached. \
                    Something else went wrong\n')

    if returnResult:
        return sol
    else:
        if flag_u0:
            return +100
        else:
            return -300 # (should be a negative error because this is probably like a model that goes too far in)


# -------------------------------- Wind --------------------------------------

def setup_globals(root,logMdot,Verbose=False,return_them=False):
    global Mdot, Edot, Ts, rs, verbose
    Mdot, Ts, verbose = 10**logMdot, 10**root[1],Verbose
    Edot = root[0]*LEdd + Mdot*c**2    # the true edot (different than old & optically thick versions)
    rs = rSonic(Ts)
    if return_them:
        return Mdot, Edot, Ts, rs, verbose


def OuterBisection(rend=1e9,tol=1e-5):

    """ Makes a full outer solution for the wind by integrating until 
    divergence, interpolating values by bissection and restarting prior to 
    divergence point, over and over until reaching rend."""

    # get the solution from the root's Ts (rs and Ts already set as global)
    if verbose: print('Calculating solution from Ts root')
    rsa,Tsa = rs,Ts
    sola = outerIntegration(r0=rsa,T0=Tsa,phi0=2.0)

    # find other solution that diverges in different direction
    if sola.status == 0: 
        sys.exit('reached end of integration interval with root!')

    elif sola.status == +1:
        direction = +1  # reached dv/dr=0,other solution needs to have higher Ts
    elif sola.status == -1:
        direction = -1 # diverged, other solution needs to have smaller Ts

    if verbose: print('Finding second solution')
    # step = 1e-6 if np.log10(Mdot)>=17 else 1e-8
    step = 5e-6
    Tsb,rsb,solb = Tsa,rsa,sola
    i=0
    while solb.status == sola.status:

        # if i>0: 
        #     Tsa,rsa,sola = Tsb,rsb,solb  
        #     # might as well update solution a 
        #     # (since this process gets closer to the right Ts)
        # Actually don't do this because it messes with the rootfinding 
        # (inward integration might no longer land on rb=RNS, but it sure does with the initial rsa because it's a root)

        logTsb = np.log10(Tsb) + direction*step
        Tsb = 10**logTsb
        rsb = rSonic(Tsb)
        solb = outerIntegration(r0=rsb,T0=Tsb,phi0=2.0)
        i+=1
        if i==20:
            print('Not able to find a solution that diverges in opposite \
                    direction after changing Ts by 20 tolerances.  \
                    Problem in the TsEdot interpolation')
            raise Exception('Should run ImproveRoot')
            # break


    # if sola was the high Ts one, switch sola and solb (just because convenient)
    if direction == -1:
        (rsa,Tsa,sola),(rsb,Tsb,solb) = (rsb,Tsb,solb),(rsa,Tsa,sola)

    if verbose:
        print('Two initial solutions. sonic point values:')
        print('logTs - sola:%.6f \t solb:%.6f'%(np.log10(Tsa),np.log10(Tsb)))
        print('logrs - sola:%.6f \t solb:%.6f'%(np.log10(rsa),np.log10(rsb)))


    def check_convergence(sola,solb,rcheck):
        """ checks if two solutions are converged (similar T, phi) at some r """
        Ta,phia = sola.sol(rcheck)
        Tb,phib = solb.sol(rcheck)
        if abs(Ta-Tb)/Ta < tol and abs(phia-phib)/phia < tol:
            return True,Ta,Tb,phia,phib
        else:
            return False,Ta,Tb,phia,phib


    # Start by finding the first point of divergence
    Npts = 1000
    R = np.logspace(np.log10(rsa),np.log10(rend),Npts)   
    # choose colder (larger) rs (rsa) as starting point because 
    # sola(rsb) doesnt exist

    # for i,ri in enumerate(R):
        # if ri>sola.t[0] and ri>solb.t[0]:
    for i,ri in enumerate(R):
        conv = check_convergence(sola,solb,ri)
        if conv[0] is False:
            i0=i            # i0 is index of first point of divergence
            break
        else:
            Ta,Tb,phia,phib = conv[1:]

    if i0==0 or i==1:
        print('Diverging at rs!')
        print(conv[1:])
        print('rs=%.5e \t rsa=%.5e \t rsb=%.5e'%(rs,rsa,rsb))
        
    # Construct initial arrays
    T,Phi = sola.sol(R[:i0])

    def update_arrays(T,Phi,sol,R,j0,jf):
        # Add new values from T and Phi using ODE solution object. 
        # Radius points to add are R[j0] and R[jf]
        Tnew,Phinew = sol(R[j0:jf+1])  # +1 so R[jf] is included
        T,Phi = np.concatenate((T,Tnew)), np.concatenate((Phi,Phinew))
        return T,Phi

    # Begin bisection
    if verbose:
        print('\nBeginning bisection')
        print('rconv (km) \t Step # \t Iter \t m')  

    # input('Pause before starting. Press enter')

    a,b = 0,1
    step,count = 0,0
    i = i0
    rconv = R[i-1]  # converged at that radius
    rcheck = R[i]   # checking if converged at that radius
    do_bisect = True
    while rconv<rend:  
        # probably can be replaced by while True if the break conditions are ok

        if do_bisect: # Calculate a new solution from interpolated values
            
            m = (a+b)/2
            Tm,phim = Ta + m*(Tb-Ta) , phia + m*(phib-phia)
            solm = outerIntegration(r0=rconv,T0=Tm,phi0=phim) 
            # go further than rmax to give it the chance to diverge either way

            if solm.status == 0: # Reached rend - done
                T,Phi = update_arrays(T,Phi,solm.sol,R,i0,Npts)  
                #jf=Npts so that it includes the last point of R
                print('reached end of integration interval  without \
                    necessarily converging.. perhaps wrong')
                return R,T,Phi

            elif solm.status == 1:
                a,sola = m,solm
            elif solm.status == -1:
                b,solb = m,solm

        else:
            i += 1
            rconv = R[i-1]
            rcheck = R[i] 

        conv = check_convergence(sola,solb,rcheck)
        if conv[0] is True:

            # Exit here if reached rend
            if rcheck==rend or i==Npts:  # both should be the same
                T,Phi = update_arrays(T,Phi,solm.sol,R,i0,i)
                return R,T,Phi

            Ta,Tb,phia,phib = conv[1:]
            a,b = 0,1 # reset bissection parameters
            step += 1 # update step counter
            count = 0 # reset iteration counter

            # Converged, so on next iteration just look further
            do_bisect = False 
        
        else:
            count+=1
            do_bisect = True

            # Before computing new solution, add converged results to array 
            # (but only if we made at least 1 step progress)
            if i-1>i0:
                T,Phi = update_arrays(T,Phi,solm.sol,R,i0,i-1)  # i-1 is where we converged last
                i0=i # next time we append

            if count==5:
                print('I seem to be a bit stuck, here are some results')
                print('sola:',sola.status,sola.t_events)
                print('solb:',solb.status,solb.t_events)
                print('conv:',conv)
                input()

        # Exit if stuck at one step
        nitermax=1000
        if count==nitermax:
            sys.exit("Could not integrate out to rend! Exiting after being \
                        stuck at the same step for %d iterations"%nitermax)

        # End of step
        if verbose and do_bisect: 
            print('%.4e \t %d \t\t %d \t\t %f'%(rconv,step,count,m))

    return R,T,Phi


def MakeWind(root, logMdot, Verbose=0, outer_only=(False,), inner_only=(False,)):
    ''' Obtaining the wind solution for set of parameters Edot/LEdd and logTs.
        If recalculating outer or inner solution, provide the initial Wind
        named tuple object in either argument (e.g. outer_only=(True,wind)) '''

    Mdot, Edot, Ts, rs, verbose = setup_globals(root,logMdot,Verbose,return_them=True)

    if verbose: 
        print('\nMaking a wind for logMdot=%.2f, logTs=%.5f, (Edot-Mdotc2)/Ledd=%.5f'
                %(logMdot,np.log10(Ts),(Edot-Mdot*c**2)/LEdd))

    # Start by finding the sonic point (it will change a bit in the first step of the bisection)
    rs = rSonic(Ts)
    
    if verbose:
        print('For log10Ts = %.2f, located sonic point at log10r = %.2f' %
              (np.log10(Ts), np.log10(rs)))


    ## Outer integration
    if not inner_only[0]:
       
        r_outer,T_outer, phi_outer = OuterBisection()
        _,rho_outer,_,_ = calculateVars_phi(r_outer, T_outer, phi=phi_outer, 
                            subsonic=False)

        # Sonic point changes slightly in the bisection process
        Tsnew,rsnew = T_outer[0], r_outer[0]
        print('Change in sonic point (caused by error in Edot-Ts relation interpolation)')
        print('root:\t Ts = %.5e \t rs = %.5e'%(Ts,rs))
        print('new:\t  Ts = %.5e \t rs = %.5e'%(Tsnew,rsnew))
        print('Judge if this is a problem or not\n')
        rs,Ts = rsnew,Tsnew

    else:
        wind = inner_only[1]
        rs = wind.rs
        isonic = list(wind.r).index(wind.rs)
        Ts = wind.T[isonic]
        r_outer,T_outer,rho_outer = wind.r[isonic:],wind.T[isonic:],wind.rho[isonic:]

    ## Inner integration
    if not outer_only[0]:

        # First inner integration
        r95 = 0.95*rs
        # r_inner1 = np.linspace(rs, r95, 500)
        r_inner1 = np.linspace(0.999*rs, r95, 30) # ignore data in 0.1% around rs
        result_inner1 = innerIntegration_r(rs,Ts)
        T95, _ = result_inner1.sol(r95)
        T_inner1, phi_inner1 = result_inner1.sol(r_inner1)

        _,rho_inner1,_,_ = calculateVars_phi(r_inner1, T_inner1, phi=phi_inner1, 
                                subsonic=True)
        rho95 = rho_inner1[-1]

        # Second inner integration 
        result_inner2 = innerIntegration_rho(rho95, T95, returnResult=True)
        rho_inner2 = np.logspace(np.log10(rho95) , np.log10(result_inner2.t[-1]), 2000)
        T_inner2, r_inner2 = result_inner2.sol(rho_inner2)
        
        print('Found base at r = %.2f km' % (r_inner2[-1]/1e5))
        rhob,Tb = rho_inner2[-1], T_inner2[-1]
        print('y=P/g= %.3e g/cm2\n'%(eos.pressure(rhob,Tb,1/3,0)/gr.grav(RNS*1e5)))


        # Attaching arrays for r,rho,T from surface to photosphere  
        #  (ignoring first point in inner2 because duplicate values at r=r95)
        r_inner = np.append(np.flip(r_inner2[1:], axis=0),
                            np.flip(r_inner1, axis=0))
        T_inner = np.append(np.flip(T_inner2[1:], axis=0),
                            np.flip(T_inner1, axis=0))
        rho_inner = np.append(np.flip(rho_inner2[1:], axis=0),
                            np.flip(rho_inner1, axis=0))
    
    else:
        wind = outer_only[1]
        isonic = list(wind.r).index(wind.rs)
        r_inner,T_inner,rho_inner = wind.r[:isonic-1],wind.T[:isonic-1],wind.rho[isonic-1]


    r = np.append(r_inner, r_outer)
    T = np.append(T_inner, T_outer)
    rho = np.append(rho_inner, rho_outer)

    # Calculate the rest of the vars
    u, rho, Lstar = calculateVars_rho(r, T, rho=rho)

    # Locate photosphere
    x = fld.x(Lstar,r,T,u)
    rph = r[np.argmin(abs(x - 0.25))]

    return IO.Wind(rs, rph, Edot, r, T, rho, u, Lstar)
 

# # For testing when making modifications to this script

# x,z = IO.load_wind_roots()
# W = MakeWind(z[0],x[0], Verbose=True)


# # Temporary
# logMdot = 17.25
# w = IO.load_wind(logMdot)
# root = IO.load_wind_roots(logMdot)
# setup_globals(root,logMdot)

# # Update rs,Ts so it matches outer solution
# rs,Ts = w.rs,w.T[list(w.r).index(w.rs)]

# print(rs/w.rs)
# print(Ts/w.T[list(w.r).index(w.rs)])

# sol = innerIntegration_r(rs,Ts)
# r_inner1 = np.linspace(0.999*rs, 0.95*rs, 30)
# T_inner1, phi_inner1 = sol.sol(r_inner1)
# _,rho_inner1,_,_ = calculateVars_phi(r_inner1, T_inner1, phi=phi_inner1, subsonic=True)

# import matplotlib.pyplot as plt
# plt.loglog(w.r,w.T,'b.')
# plt.loglog(r_inner1,T_inner1,'r.')
# plt.show()