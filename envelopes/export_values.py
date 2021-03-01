import sys
sys.path.append(".")
sys.path.append("./analysis")

import numpy as np
import IO

from scipy.interpolate import interp1d
from scipy.integrate import quad

import physics
params  = IO.load_params()
eos = physics.EOS_FLD(params['comp'])

# Load stellar params and methods
if IO.load_params()['FLD'] == True:
    from env_GR_FLD import Swz
else:
    from env_GR import Swz


def export(target = "."):

    # Export useful values for analysis for each Rphot to a text file at target directory
    # Current values are : Linf,Rb,Tb,Rhob,Pb,Tphot,Rhophot
    # tsound(sound crossing time) 
    # Min&Mout (masses below & above sonic point)

    if target[-1]!='/': target += '/'
    Rphotkms = IO.get_phot_list()
    filename = target+'env_values_'+IO.get_name()+'.txt'

    with open(filename,'w') as f:

        f.write(('{:<11s} \t '*11 +'{:<11s}\n').format(
            'Rph (km)','Linf (erg/s)','rb (cm)','Tb (K)','rhob (g/cm3)','Pb (dyne/cm2)','Tph (K)','rhoph (g/cm3)','tsound (s)','E_env (erg)','t_thermal (s)','Min (g)'))


        for R in Rphotkms:

            # if R>=params['R']+0.1: # not the ultra compact ones
            if True:

                print(R)
                env = IO.read_from_file(R)
                
                cs = np.sqrt(eos.cs2(env.T))
                func_inverse_cs = interp1d(env.r,1/cs,kind='cubic')
                tsound,err = quad(func_inverse_cs,env.r[0],env.rphot,epsrel=1e-5)
                E,tau_th = thermaltime(env.r,env.rho,env.T,env.Linf,env.rphot)

        
                # Mass contained in envelope
                rhofunc = interp1d(env.r,env.rho,kind='cubic')

                def mass_in_shell(r): 
                    return 4*np.pi*rhofunc(r)*r**2

                r0 = params['R']*1e5 + 2e2 # start integrating 2m above surface to make uniform
                Min,err = quad(mass_in_shell, r0, env.rphot, epsrel=1e-5, limit=500)

                # Write base values
                f.write(('%0.1f \t\t' + '%0.6e \t'*5)%
                    (R,env.Linf,env.r[0],env.T[0],env.rho[0],eos.pressure(env.rho[0],env.T[0],lam=1/3,R=0)))
                
                # Write photoshere values 
                iphot = list(env.r).index(R*1e5) if params['FLD'] == True else -1
                f.write(('%0.6e \t'*2)%
                    (env.T[iphot],env.rho[iphot]))

                # Timescales
                f.write(('%0.6e \t'*3)%
                    (tsound,E,tau_th))

                # Mass contained
                f.write(('%0.6e \t')%
                    (Min))

                f.write('\n')

    print('Saved values to : "%s"'%filename)



def thermaltime(r,rho,T,Linf,rphot):
    '''Thermal time defined as energy contained in envelope (radiation, gravitational,
        kinetic) divided by luminosity at the base'''

    fT = interp1d(r,T,kind='linear')
    frho = interp1d(r,rho,kind='linear')

    arad = 7.5657e-15
    c = 2.99792458e10

    def dE(ri):  # From Fowler 1964
        u = arad*fT(ri)**4
        x = u*Swz(ri)**(-0.5) + (frho(ri)+u/c**2)*c**2*(1-Swz(ri)**(-0.5))
        dV = 4*np.pi*ri**2
        return x*dV

    E = abs(quad(dE,r[0],rphot,epsrel=1e-5)[0])
    tau_th = E/Linf

    return E,tau_th


# Command line call
if len(sys.argv)>=2:
    export(target = sys.argv[-1])
else:
    export()