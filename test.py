# Load models for M=1.4 Msun, R=12 km and reproduce Fig. 8 (top 2 panels)

import numpy as np
import matplotlib.pyplot as plt
from IO import get_wind_list,read_wind,get_envelope_list

LEdd = 4*np.pi*2.99792458e10*6.6726e-8*1.4*2e33/0.2

def Make_figure():

    fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
    fig.subplots_adjust(hspace=0)
    ax1.set_ylabel(r'T ($10^9$ K)')
    ax2.set_ylabel(r'r (km)')
    ax2.set_xlabel(r'$L_b^{\infty}/L_\mathrm{Edd}$')

    # Winds
    logMdots = get_wind_list()
    Lbinf_wind,Tb_wind,rph_wind,rs_wind = [[] for i in range(4)]
    for logMdot in [x for x in logMdots if x<=18.5]:
        w = read_wind(logMdot)
        Lbinf_wind.append(w.Lstar[0])
        Tb_wind.append(w.T[0])
        rph_wind.append(w.rph)
        rs_wind.append(w.rs)

    Lbinf_wind,Tb_wind,rph_wind,rs_wind = [np.array(l) for l in (Lbinf_wind,Tb_wind,rph_wind,rs_wind)]

    ax1.plot(Lbinf_wind/LEdd, Tb_wind/1e9, color='tab:blue', ls='-', lw=0.8)
    ax2.semilogy(Lbinf_wind/LEdd, rph_wind/1e5, color='tab:blue', ls='-', lw=0.8)
    ax2.semilogy(Lbinf_wind/LEdd, rs_wind/1e5, color='tab:blue', ls='-.', lw=0.8)


    # Envelopes
    # rph_env = get_envelope_list()
    # Linf_env,Tb_env = [[] for i in range(2)]





    ax1.axvline(1,color='k',ls='--',lw=0.6)
    ax2.axvline(1,color='k',ls='--',lw=0.6)

    plt.show()

Make_figure()
        
