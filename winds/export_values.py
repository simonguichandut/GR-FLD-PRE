import sys
sys.path.append(".")
sys.path.append("./analysis")

import numpy as np
import IO
from timescales import *


def export(target = "."):

    # Export useful values for analysis for each Mdot to a text file at target directory
    # Current values are : Tb,Rhob,Pb,Lb,Lb*,Rphot,Tphot,Rhophot,Lphot,Lphot*,rs,
    # tsound (sound crossing time), tsound2 (rs/cs(rs)), tflow (flow crossing time), Tau : a specific timescale, currently : sound crossing time until v=1e6, then flow crossing time
    # Min&Mout (masses below & above sonic point)

    if target[-1]!='/': target += '/'
    logMDOTS,_ = IO.load_roots()
    params  = IO.load_params()
    filename = target+'wind_values_'+IO.get_name()+'.txt'

    from scipy.interpolate import interp1d
    from scipy.integrate import quad

    with open(filename,'w') as f:

        f.write(('{:<11s} \t '*20 +'{:<11s}\n').format(
            'logMdot (g/s)','rb (cm)','Tb (K)','rhob (g/cm3)','Pb (dyne/cm2)','Lb (erg/s)','Lb* (erg/s)','Rph (cm)','Tph (K)','rhoph (g/cm3)','Lph (erg/s)','Lph* (erg/s)','rs (cm)','tflow (s)','tsound (s)','tsound2 (s)','Tau (s)','E_env (erg)','t_thermal (s)','Min (g)','Mout (g)'))


        for logMdot in logMDOTS[::-1]:
            print(logMdot)

            if IO.info(logMdot,returnit=True)['datafile_exists'] is True:

                w = IO.read_from_file(logMdot) 

                if params['FLD'] == True:
                    from photosphere import Rphot_tau_twothirds,Rphot_Teff
                    # rph = Rphot_tau_twothirds(logMdot)
                    rph  = Rphot_Teff(logMdot)
                    iph = np.argwhere(w.r==rph)[0][0]
                else:
                    rph = w.r[-1]
                    iph = -1

                tflow = flowtime(w.r,w.u,rphot=rph)
                tsound = soundtime(w.r,w.cs,rphot=rph)
                tsound2 = soundtime2(w.r,w.cs,w.rs)
                Tau = soundflowtime(w.r,w.cs,w.u,w.rs,rphot=rph)
                E,tau_th = thermaltime(w.r,w.rho,w.T,w.u,w.Lstar[0],rphot=rph)

                # Mass above and below sonic point
                rhofunc = interp1d(w.r,w.rho,kind='cubic')

                def mass_in_shell(r): 
                    return 4*np.pi*rhofunc(r)*r**2

                r0 = params['R']*1e5 + 1e3 # start integrating 10m above surface to make uniform
                if w.r[0]>r0:
                    print('Warning: rbase quite large (%.3f)'%(w.r[0]/1e5))
                    r0=w.r[0]

                Min,_ = quad(mass_in_shell, r0, w.rs, epsrel=1e-5, limit=500)
                if params['FLD'] == True:
                    Mout,_ = quad(mass_in_shell, w.rs , rph, epsrel=1e-5, limit=500)
                else:
                    Mout,_ = quad(mass_in_shell, w.rs , w.r[-1], epsrel=1e-5, limit=500)

                # print(Min/Mout)

                # f.write(('%0.2f \t\t '+'%0.6e \t '*15 + '%0.6e\n')%
                #     (x,w.r[0],w.T[0],w.rho[0],w.P[0],w.L[0],w.Lstar[0],w.r[-1],T[-1],rho[-1],L[-1],Lstar[-1],rs,tflow,tsound,tsound2,Tau))


                # Write base values
                f.write(('%0.2f \t\t' + '%0.6e \t'*6)%
                    (logMdot,w.r[0],w.T[0],w.rho[0],w.P[0],w.L[0],w.Lstar[0]))
                
                # Write photoshere values 
                f.write(('%0.6e \t'*5)%
                    (rph,w.T[iph],w.rho[iph],w.L[iph],w.Lstar[iph]))

                # sonic point + timescales
                f.write(('%0.6e \t'*7)%
                    (w.rs,tflow,tsound,tsound2,Tau,E,tau_th))

                # Mass contained
                f.write(('%0.6e \t'*2)%
                    (Min,Mout))

                f.write('\n')

    print('Saved values to : "%s"'%filename)

# Command line call
if len(sys.argv)>=2:
    export(target = sys.argv[-1])
else:
    export()