import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple, HandlerBase
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from matplotlib.ticker import (FixedLocator,LogLocator,NullFormatter)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

mpl.rcParams.update({

    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # Non-italic math
    "mathtext.default": "regular",
    # Tick settings
    "xtick.direction" : "in",
    "ytick.direction" : "in",
    "xtick.top" : True,
    "ytick.right" : True,
    # Short dash sign
    "axes.unicode_minus" : True
})


def set_size(width, fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    elif width == 'mnras2col':
        width_pt = 240
    elif width == 'mnras1col':
        width_pt = 504
    else:
        width_pt = width
    # Width of figure
    fig_width_pt = width_pt * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim



# Main model
import IO
IO.change_param('M',1.4,print_update=False)
IO.change_param('R',12,print_update=False)
IO.change_param('comp','He',print_update=False)
IO.change_param('yb',1e8,print_update=False)
# IO.change_param('Prad','exact',print_update=True)
IO.change_param('Prad','exact',print_update=False)

import physics
from winds.wind_GR_FLD import *
from envelopes.env_GR_FLD import *


img = 'pdf'
dpi = 300 if img == 'png' else None



# ------------------------------------------------ WIND PLOTS ------------------------------------------------

def Make_lineshift_plot(figsize):

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(figsize[0],1.8*figsize[1]))
    fig.subplots_adjust(hspace=0.4)

    ax1.set_xlabel(r'$r$ (km)')
    ax1.set_ylabel(r'$\Delta\lambda/\lambda$')
    w = IO.load_wind(18)
    blue = np.sqrt((1-w.u/c)/(1+w.u/c))
    red = gr.Swz(w.r)**-0.5
    ax1.plot(w.r/1e5,red-1,'r-',lw=0.8, label='redshift')
    ax1.plot(w.r/1e5,blue-1,'b-',lw=0.8, label='blueshift')
    ax1.plot(w.r/1e5,red*blue-1,'k-',lw=0.8, label='total')
    ax1.legend(frameon=False,ncol=3,bbox_to_anchor=(0,0.95,1,0.95),bbox_transform=ax1.transAxes,loc='lower left',mode='expand',borderaxespad=1)

    ax1.axvline(w.rs/1e5,color='k',ls='--',lw=0.7)
    ax1.text(w.rs/1e5+5,0.085,r'$r_s$',ha='left',va='center')
    ax1.set_ylim([-0.02,0.1])
    ax1.set_xlim([-10,500])

    ax1.text(0.98,0.9,r'$\log\dot{M}$=18 model',fontsize=8,fontweight='bold',transform=ax1.transAxes,ha='right',va='center')

    ax2.set_xlabel(r'$\log\dot{M}$ (g s$^{-1}$)')
    ax2.set_ylabel(r'$\Delta\lambda/\lambda$')

    logMdots = IO.get_wind_list()
    logMdots  = logMdots[:logMdots.index(18.5)+1] # ignore logMdot>18.5
    blue,red = [],[]
    logMdots2 = []

    for logMdot in logMdots:

        logMdots2.append(logMdot)

        w = IO.load_wind(logMdot)
        rph = w.rph

        if logMdot==18:
            ax1.axvline(rph/1e5,color='k',ls='--',lw=0.7)
            ax1.text(rph/1e5-10,0.085,r'$r_\mathrm{ph}$',ha='right',va='center')


        vph = w.u[list(w.r).index(rph)]
        blue.append(np.sqrt((1-vph/c)/(1+vph/c)))
        red.append(gr.Swz(rph)**-0.5)

    blue,red = np.array(blue),np.array(red)

    ax2.plot(logMdots2,red-1,'r-',lw=0.8,label='redshift')
    ax2.plot(logMdots2,blue-1,'b-',lw=0.8,label='blueshift')
    ax2.plot(logMdots2,blue*red-1,'k-',lw=0.8,label='total')
    ax2.set_yticks([-0.005,0,0.005,0.01,0.015,0.02])
    ax2.set_ylim([-0.008,0.023])

    ax2.text(0.98,0.9,r'All models, values at $r_\mathrm{ph}$',fontsize=8,fontweight='bold',transform=ax2.transAxes,ha='right',va='center')
    ax1.axhline(0,color='k',ls=':',lw=0.7)
    ax2.axhline(0,color='k',ls=':',lw=0.7)


    fig.savefig('paper_plots/wind_lineshift.'+img,bbox_inches='tight',format=img,dpi=dpi)
    print('Saved figure to paper_plots/wind_lineshift.'+img)


def Make_wind_paramspace_plot(figsize):

    fig,ax = plt.subplots(1,1,figsize=(figsize[0],figsize[0])) # square plot

    ax.set_xlabel(r'$\log r_s$')
    ax.set_ylabel(r'$(\dot{E}-\dot{M}c^2)/L_\mathrm{Edd}$')

    # We use the non-exact Prad models for this plot because the EdotTs rel lines are more complete
    logMdots,roots = IO.load_wind_roots(specific_file='winds/roots/roots_He_M1.4_R12_y8.txt')
    logMdots,roots = logMdots[:logMdots.index(18.5)+1], roots[:logMdots.index(18.5)+1]

    def logTs_to_logrs(logMdot,Edot_LEdd,logTs):
        setup_globals([Edot_LEdd,logTs],logMdot)
        return np.log10(rSonic(10**logTs))

    Edotvals = [root[0] for root in roots]
    logTsvals = [root[1] for root in roots]
    logrsvals = [ logTs_to_logrs(logMdot,Edot,logTs) for logMdot,Edot,logTs in zip(logMdots,Edotvals,logTsvals)]

    ax.plot(logrsvals,Edotvals,color='tab:blue',ls='-',lw=1)
    ax.set_xlim([6,8])
    ax.set_ylim([1.012,1.03])
    ax.set_yticks(np.arange(1.012,1.03,0.003))

    # where to write the logMdot text for the logMdots to highlight
    xpos = (6.53,6.12,6.1,6.05,6.25,6.5,6.8)
    ypos = (1.0127,1.015,1.0185,1.025,1.029,1.029,1.029)

    j=0
    for i,logMdot in enumerate(logMdots):

        _,rel_Edot,rel_Ts,_ = IO.load_EdotTsrel(logMdot, specific_file='winds/roots/rel/He_M1.4_R12/EdotTsrel_%.2f.txt'%logMdot)
        rel_rs = [logTs_to_logrs(logMdot,Edot,logTs) for Edot,logTs in zip(rel_Edot,rel_Ts)]

        if logMdot in (17,17.25,17.5,17.75,18,18.25,18.5):

            ax.plot(rel_rs,rel_Edot,'k-',lw=0.9)
            ax.plot(logrsvals[i],[Edotvals[i]],'o',mfc='w',mec='k',ms=3)

            if logMdot==17.25:
                s,fs = (r'log$\dot{M}$='+str(logMdot)),8
            else:
                s,fs = str(logMdot),7

            # ax.text(Edotvals[i]+0.0005,logrsvals[i],s,fontsize=8,transform=ax.transData,ha='left',va='center')
            ax.text(xpos[j],ypos[j],s,fontsize=fs,transform=ax.transData,ha='left',va='center')
            j+=1

        else:
            if len(str(logMdot))==4 or str(logMdot)[-1] in ('0','5'): # don't plot e.g. 17.06
                ax.plot(rel_rs,rel_Edot,'k-',lw=0.3,alpha=0.4)

    fig.savefig('paper_plots/wind_paramspace.'+img,bbox_inches='tight',format=img,dpi=dpi)
    print('Saved figure to paper_plots/wind_paramspace.'+img)


def Make_base_Enuc_plot(figsize):

    fig,ax = plt.subplots(1,1,figsize=figsize)
    ax.set_xlabel(r'$\log E_\mathrm{nuc}$ (erg g$^{-1}$)')
    ax.set_ylabel(r'$\log\dot{M}$ (g s$^{-1}$)')

    logMdots = IO.get_wind_list()
    beta_base = []
    logMdots2 = []
    for logMdot in logMdots:
        if logMdot<=18.75:
            w = IO.load_wind(logMdot)
            beta_base.append(eos.Beta(w.rho[0],w.T[0],lam=1/3,R=0))
            logMdots2.append(logMdot)

    # Necessary values of Enuc
    GM,R = 6.6726e-8*2e33*1.4, 12e5
    g = GM/R**2 * (1-2*GM/c**2/R)**(-0.5)
    mu = 4/3
    yb = 1e8
    fEnuc = lambda alpha : 0.5 * kB/mu/mp * (3*g*yb/arad)**0.25 * alpha**0.25*(3*alpha+5)/(1-alpha)

    # loglog
    ax.plot(np.log10(fEnuc(1-np.array(beta_base))), logMdots2,'k-',lw=0.7)

    ax.axvline(np.log10(0.6*1.602177e-6/mp),linestyle='--',color='r',lw=0.7)
    ax.text(17.8,16.9,r'0.6',ha='left',va='center',color='r',fontsize=7)
    ax.axvline(np.log10(1.6*1.602177e-6/mp),linestyle='--',color='r',lw=0.7)
    ax.text(18.25,16.9,r'1.6 MeV/nucleon',ha='left',va='center',color='r',fontsize=7)

    # Add custom ticks on the right axis
    ax2 = ax.twinx()
    ax2.plot(np.log10(fEnuc(1-np.array(beta_base))), logMdots2, alpha=0) # same line to get the same scaling

    flogMdot = IUS(beta_base[::-1],logMdots2[::-1])

    ax2.set_ylabel(r'$\beta_b$')
    beta_vals = [0.01,0.1,0.3,0.5,0.57]
    ax2.set_yticks([flogMdot(beta) for beta in beta_vals])
    ax2.set_yticklabels([str(beta) for beta in beta_vals])


    # TICKS
    x_major = FixedLocator([18,18.5,19,19.5])
    ax.xaxis.set_major_locator(x_major)
    x_minor = FixedLocator([17.5,17.6,17.7,17.8,17.9,18.1,18.2,18.3,18.4,18.6,18.7,18.8,18.8,19.1,19.2,19.3,19.4,19.6,19.7,19.8,19.9])
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(NullFormatter())


    fig.savefig('paper_plots/wind_base_Enuc.'+img,bbox_inches='tight',format=img,dpi=dpi)
    print('Saved figure to paper_plots/wind_base_Enuc.'+img)


def Make_Mdot_prescription_plot(figsize):

    fig,ax = plt.subplots(1,1,figsize=figsize)
    ax.set_xlabel(r'$L_b^\infty/L_\mathrm{Edd}$')
    ax.set_ylabel(r'$\xi$')

    logMdots = IO.get_wind_list()
    logMdots2,Lbs = [],[]

    enth_inf1, enth_inf2 = [],[] # enthalpy at infinity

    for logMdot in logMdots:
        if logMdot>16.75 and logMdot<18.5:
            logMdots2.append(logMdot)
            w = IO.load_wind(logMdot)
            Lbs.append(w.Lstar[0])

            enth_inf1.append(2*LEdd/10**logMdot*w.u[-1]/c)
            enth_inf2.append(2*arad*w.T[-1]**4/w.rho[-1])


    Lbs = np.array(Lbs)
    # PP86 prediction
    Mdots_pred = (Lbs-LEdd)/(6.6726e-8*2e33*1.4/12e5)


    # or plot as ratio
    ax.plot(Lbs/LEdd, Mdots_pred/10**np.array(logMdots2), 'k-', label='model values', lw=0.7)

    ## plot more correct formula with redshit and enthalpy at infinity
    # corr1 = c**2*12e5/GM * (1 - gr.Swz(12e5)**0.5 + np.array(enth_inf1)/c**2)
    # ax.plot(Lbs/LEdd, corr1, 'g-', lw=0.8, label=r'$\xi=\frac{c^2R}{GM}\left[1-\left(1-\frac{2GM}{c^2R}\right)^{1/2}+\frac{w_\infty}{c^2}\right]$')

    ax.axhline(1,color='k',ls=':',lw=0.8, label=r'$\xi=1$')
    ax.set_ylim([0.95,1.6])

    # ax.legend(frameon=False)

    fig.savefig('paper_plots/Mdot_prescription.'+img,bbox_inches='tight',format=img,dpi=dpi)
    print('Saved figure to paper_plots/Mdot_prescription.'+img)


# ---------------------------------------------- ENVELOPE PLOTS ----------------------------------------------

def Make_touchdown_error_plot(figsize):

    # It's only here that we'll have the very compact envelopes
    keV = 1.602177e-9

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(figsize[0],1.5*figsize[1]),sharex=True)
    fig.subplots_adjust(hspace=0)

    ax2.set_xlabel(r'$r_\mathrm{ph}-R$ (cm)')
    ax1.set_ylabel(r'$L^\infty/L_\mathrm{Edd}$')
    ax2.set_ylabel(r'$kT^\infty_\mathrm{eff}$ (keV)')

    rphots0 = IO.get_envelope_list()
    Lratio,pavlov_f,Teffinf = [],[],[]

    rphots = [] # those that will be plotted
    for rphotkm in rphots0:

        if rphotkm >= 12.005:

            env = IO.load_envelope(rphotkm)

            if env.Linf/LEdd>=1:
                break

            rphots.append(rphotkm)

            Lb = env.Linf*gr.Swz(env.r[0])**(-1)
            Lcrb = gr.Lcrit(env.r[0],env.rho[0],env.T[0],eos)

            Lratio.append(env.Linf/LEdd)
            pavlov_f.append(Lb/Lcrb)

            iphot = list(env.r).index(env.rph)
            Teffinf.append(env.T[iphot]*gr.Swz(rphotkm*1e5)**(+0.5))

    rphots,Teffinf = np.array(rphots), np.array(Teffinf)

    ax1.semilogx((rphots-RNS)*1e5, Lratio, 'k-', lw=0.8)
    ax2.semilogx((rphots-RNS)*1e5, kB*Teffinf/keV, 'k-', lw=0.8)

    # Pavlov color correction
    def fc_pavlov(x):  # x=Linf/LEdd (might have to change this if I misunderstand Pavlov)
        return (0.14*np.log(3/(1-x)) + 0.59)**(-4/5) * (3/(1-x))**(2/15) * x**(3/20)

    fc = np.array([fc_pavlov(x) for x in pavlov_f])
    ax2.semilogx((rphots-RNS)*1e5, fc*kB*Teffinf/keV, 'k--', lw=0.8, label=(r'$T_\mathrm{color}$ (Pavlov et al. 1991)'))
    ax2.legend(loc='lower center', frameon=False)


    # # Show specific example mapping between % error in kT with % error on NS radius. Take point which is closest to 1.8 keV
    # Teff_interp = IUS(rphots,Teffinf)
    # rvec = np.linspace(rphots[0],rphots[-1],200)
    # Tvec = Teff_interp(rvec)
    # Teff_obs = Tvec[np.argmin(abs(kB*Tvec/keV-1.8))]
    # i = list(Tvec).index(Teff_obs)
    # Robs = rvec[i]
    # ybounds = [Teff_obs - 0.1*keV/kB, Teff_obs + 0.1*keV/kB]

    # rvec_left = np.linspace(rphots[0],rphots[i],100)
    # rvec_right = np.linspace(rphots[i],rphots[-1],100)
    # bound_left = np.argmin(abs(Teff_interp(rvec_left)-ybounds[0]))
    # bound_right = np.argmin(abs(Teff_interp(rvec_right)-ybounds[0]))
    # Rlow, Rhigh = rvec_left[bound_left], rvec_right[bound_right]
    # # print(kB*Teff_obs/keV,kB*ybounds[0]/keV,(rvec_left[bound_left]-RNS)*1e5,(rvec_right[bound_right]-RNS)*1e5)
    # print('If the measured temperature if %.3f +- 0.1 keV, the inferred radius is %.3f + %.3f - %.3f km'%
    #         (kB*Teff_obs/keV, Robs, Rhigh-Robs, Robs-Rlow ))
    # print('Or an uncertainty of %.2f %%'%((Rhigh-Rlow)/Robs*100))

    # Uncomment to plot errorbars
    # ax2.plot([(Rlow-RNS)*1e5, (Rhigh-RNS)*1e5], [kB*ybounds[0]/keV,kB*ybounds[0]/keV],'r--', lw=0.6)
    # ax2.errorbar([(Robs-RNS)*1e5],[kB*Teff_obs/keV],yerr=0.05*kB*Teff_obs/keV,ecolor='r',elinewidth=0.7,capsize=3)
    # diff = Rhigh-Rlow
    # ax2.errorbar([(Rlow+diff/2-RNS)*1e5],[kB*ybounds[0]/keV],xerr=diff/2*1e5,ecolor='r',elinewidth=0.7,capsize=3)

    # Add the newtonian envelopes
    from envelopes.newtonian import find_rph, find_rph_x1
    x = np.logspace(-2,0,500)  # L/LEdd
    rphot_newt = []
    for xi in x[:-1]: # remove the exactly 1 value
        rphot_newt.append(find_rph(xi))

    rphot_newt.append(find_rph_x1()) # add the x=1 value

    rphot_newt = np.array(rphot_newt)

    ax1.semilogx(rphot_newt - RNS*1e5, x, 'k:', lw=0.8, label='Newtonian models')
    ax1.legend(loc='lower center', frameon=False)

    fig.savefig('paper_plots/env_touchdown.'+img,bbox_inches='tight',format=img,dpi=dpi)
    print('Saved figure to paper_plots/env_touchdown.'+img)


def Make_env_paramspace_plot(figsize):

    #  where (x,y) are (log(1-Lph/Lcrit) , rho_ph/rho_2/3 (rho_2/3 is ) OR log10(rho_ph)

    fig,ax = plt.subplots(1,1,figsize=(figsize[0],figsize[0])) # square plot

    ax.set_xlabel(r'$\log(1-L_\mathrm{ph}/L_\mathrm{cr})$')
    ax.set_ylabel(r'$\log\rho_\mathrm{ph}$')

    ax.set_xlim([-4.15,-3.75])
    ax.set_ylim([-5.5,-4.1])

    f0_root,rhoph_root = [],[]

    rphots = IO.get_envelope_list()
    rphots = rphots[rphots.index(12.1):] # 12.1km and up

    for rph in rphots:
        env = IO.load_envelope(rph)
        rhoph_root.append(env.rho[list(env.r).index(env.rph)])

        Tph = env.T[list(env.r).index(env.rph)]
        f0 = get_f0(env.rph, Tph, env.Linf)
        f0_root.append(f0)

    ax.plot(f0_root, np.log10(rhoph_root), color='tab:red',ls='-',lw=1)

    # where to write the rph text for the rph curves to highlight
    xpos = (-4.1,-4.1,-4.1,-4.1,-4.1,-4.1)
    ypos = (-4.3,-4.41,-4.65,-4.92,-5.12,-5.3)
    j=0

    def load_rhophf0rel_TEMP(Rphotkm):

        s = str(Rphotkm)
        if s[-2:]=='.0': s=s[:-2]

        name = IO.get_name(include_Prad=False)
        i = name.find('y')
        name = name[:i-1] + name[i+2:]
        path = 'envelopes/roots/' + name

        filepath = 'envelopes/roots/' + name + '/rhophf0rel_' + s + '.txt'

        f0, rhophA, rhophB = [],[],[]
        with open(filepath,'r') as f:
            next(f)
            for line in f:
                f0.append(eval(line.split()[0]))
                rhophA.append(10**eval(line.split()[1]))
                rhophB.append(10**eval(line.split()[2])) # saved as log values in the file

        return True,f0,rhophA,rhophB


    for i,rph in enumerate(rphots):

        _,f0vals,rhophvals,_ = IO.load_rhophf0rel(rph)
        rhophvals = np.array(rhophvals)

        if rph in (13,15,20,30,40,50):
            ax.plot(f0vals, np.log10(rhophvals),'k-',lw=0.9)
            ax.plot([f0_root[i]],np.log10(rhoph_root[i]),'o',mfc='w',mec='k',ms=4)

            if rph==50:
                s,fs = (r'$r_\mathrm{ph}$='+str(int(rph))),8
            else:
                s,fs = str(int(rph)),7

            ax.text(xpos[j],ypos[j],s,fontsize=fs,transform=ax.transData,ha='center',va='center')
            j+=1

        else:
            ax.plot(f0vals, np.log10(rhophvals),'k-', lw=0.3, alpha=0.4)


    fig.savefig('paper_plots/env_paramspace.'+img,bbox_inches='tight',format=img,dpi=dpi)
    print('Saved figure to paper_plots/env_paramspace.'+img)


def Make_env_energy_plot(figsize):

    fig,ax2 = plt.subplots(1,1,figsize=figsize)
    ax2.set_xlabel('$m$ (g)')
    ax2.set_ylabel('$\Delta E/m$ (erg g$^{-1}$)')

    def energies(r,rho,T,rphot,npts=1000):

        frho,fT = IUS(r,rho),IUS(r,T)

        def dE_rad(ri):
            return arad*fT(ri)**4 * gr.Swz(ri)**(-1/2) * 4*np.pi*ri**2
        def dE_gas(ri):
            return 1.5*kB*fT(ri)*frho(ri)/eos.mu/mp * 4*np.pi*ri**2
        def dE_grav(ri):
            return frho(ri)*c**2*(1-gr.Swz(ri)**(-0.5)) * 4*np.pi*ri**2

        rvec = np.logspace(np.log10(r[0]) , np.log10(rphot), npts)

        E_rad,E_gas,E_grav = [],[],[]
        for ri in rvec:
            for y,func in zip((E_rad,E_gas,E_grav) , (dE_rad,dE_gas,dE_grav)):
                y.append(quad(func,ri,rphot,epsrel=1e-5)[0])

        E_rad,E_gas,E_grav = [np.array(y) for y in (E_rad,E_gas,E_grav)]
        E_tot = E_rad+E_gas+E_grav

        return rvec,E_rad,E_gas,E_grav,E_tot


    def mass(r,rho,rphot,npts=1000):
        frho = IUS(r,rho)

        def dm(ri):
            return frho(ri)*4*np.pi*ri**2

        rvec = np.logspace(np.log10(r[0]) , np.log10(rphot), npts)

        M = []
        for ri in rvec:
            M.append(quad(dm,ri,rphot,epsrel=1e-5)[0])

        return rvec, M


    for Rphotkm in (12.005,12.01,12.02,12.05,12.1,13,30):

        env = IO.load_envelope(Rphotkm)

        npts = 2000
        rvec,E_rad,E_gas,E_grav,E_tot = energies(env.r,env.rho,env.T,env.rph,npts=npts)

        rvecb,M = mass(env.r,env.rho,env.rph,npts=npts)

        if Rphotkm == 12.005:
            fE0 = IUS(M[::-1],E_tot[::-1])
            Mvec = np.logspace(12,np.log10(max(M)),500)

        else:
            fE = IUS(M[::-1],E_tot[::-1])
            deltaE = fE(Mvec) - fE0(Mvec)

            ax2.loglog(Mvec,deltaE/Mvec,lw=0.5,label=('%.2f'%Rphotkm))


    lg = ax2.legend(frameon=False,fontsize=7,title=r'$r_\mathrm{ph}$ (km)',loc=1)
    lg.get_title().set_fontsize(7)
    ax2.set_xlim([7e12,4e21])
    ax2.set_ylim([5e16,3e20])

    ## TICKS
    x_major = LogLocator(base = 10.0, numticks = 4)
    ax2.xaxis.set_major_locator(x_major)
    x_minor = LogLocator(base = 10.0,numticks=20)
    ax2.xaxis.set_minor_locator(x_minor)
    ax2.xaxis.set_minor_formatter(NullFormatter())

    fig.savefig('paper_plots/env_energy.'+img,bbox_inches='tight',format=img,dpi=dpi)
    print('Saved figure to paper_plots/env_energy.'+img)


# ----------------------------------------------- COMPARE PLOTS -----------------------------------------------

def Make_profiles_plot(figsize):

    fig,((ax1a,ax1c),(ax1b,ax1d)) = plt.subplots(2,2,figsize=figsize,sharex=True)
    fig.subplots_adjust(hspace=0,wspace=0.25)

    ax1a.set_ylabel(r'$T$ (K)')
    ax1b.set_ylabel(r'$\rho$ (g cm$^{-3}$)')

    ax1a.set_xlim([1e6,1e9])
    ax1a.set_ylim([6e5,2e9])
    rthin=np.logspace(7.2,9,100)
    Tthin=(LEdd/(4*np.pi*rthin**2*arad*c))**0.25

    ax1b.set_ylim([1e-9,2e5])
    ax1b.set_xlabel(r'r (cm)')

    ax1c.set_ylabel(r'v (cm s$^{-1}$)')
    ax1c.set_ylim([8e2,7e8])

    ax1d.set_ylabel(r'$x=F/cU_R$')
    ax1d.set_xlabel(r'r (cm)')
    ax1d.set_ylim([1e-7,5])

    ax1d.axhline(1,color='k',ls='--',lw=1.3)

    # Winds
    # print('Wind models in profiles.'+img)
    for i,logMdot in enumerate([17.25,17.5,17.75,18,18.25,18.5]):

        w = IO.load_wind(logMdot)
        if i==0: w0=w # save first wind for arrow

        iphot = list(w.r).index(w.rph)
        isonic = list(w.r).index(w.rs)

        label = 'winds' if i==0 else None

        R = np.array(w.r)
        x = fld.x(w.Lstar,w.r,w.T,w.u)

        ax1a.loglog(R,w.T,ls='-',color='tab:blue',lw=0.5,label=label)
        ax1b.loglog(R,w.rho,ls='-',color='tab:blue',lw=0.5)
        ax1c.loglog(R,w.u,ls='-',color='tab:blue',lw=0.5)
        ax1d.loglog(R,x,ls='-',color='tab:blue',lw=0.5)

        ax1a.loglog(R[isonic],w.T[isonic],marker='x',color='tab:blue',ms=2)
        ax1b.loglog(R[isonic],w.rho[isonic],marker='x',color='tab:blue',ms=2)
        ax1c.loglog(R[isonic],w.u[isonic],marker='x',color='tab:blue',ms=2)
        ax1d.loglog(R[isonic],x[isonic],marker='x',color='tab:blue',ms=2)

        ax1a.loglog(R[iphot],w.T[iphot],marker='.',color='tab:blue',ms=3)
        ax1b.loglog(R[iphot],w.rho[iphot],marker='.',color='tab:blue',ms=3)
        ax1c.loglog(R[iphot],w.u[iphot],marker='.',color='tab:blue',ms=3)
        ax1d.loglog(R[iphot],x[iphot],marker='.',color='tab:blue',ms=3)

        # print('logMdot=%.2f \t Lbinf/LEdd=%.2f'%(logMdot,Lbinf_wind[Mdotindex]/LEdd))


    # Envelopes
    Rphotkms = (13,15,20,30,40,50)
    # print('Envelope models in profiles.'+img)
    for i,Rphotkm in enumerate(Rphotkms):

        env = IO.load_envelope(Rphotkm)
        if i==0: env0=env # save first env for arrow

        iphot = list(env.r).index(env.rph)

        label = 'envelopes' if i==0 else None

        R = np.array(env.r)
        x = fld.x(env.Linf,env.r,env.T,v=0)

        ax1a.loglog(R,env.T,ls='-',color='tab:red',lw=0.5,label=label)
        ax1b.loglog(R,env.rho,ls='-',color='tab:red',lw=0.5)
        ax1d.loglog(R,x,ls='-',color='tab:red',lw=0.5)

        ax1a.loglog(R[iphot],env.T[iphot],marker='.',color='tab:red',ms=3)
        ax1b.loglog(R[iphot],env.rho[iphot],marker='.',color='tab:red',ms=3)
        ax1d.loglog(R[iphot],x[iphot],marker='.',color='tab:red',ms=3)

        # print('Rphot=%.2f \t Linf/LEdd=%.2f'%(Rphotkm,env.Linf/LEdd))

    # Arrows and annotations
    iw = np.argmin(abs(w0.r-3e6))
    # ax1a.annotate('',xy=(7e6,1e8),xytext=(w0.r[iw],w0.T[iw]),arrowprops=dict(color='tab:blue',arrowstyle='->',linewidth=0.8,mutation_scale=7))
    ax1a.annotate('',xy=(7e6,1e8),xytext=(3e6,2.6e7),arrowprops=dict(color='tab:blue',arrowstyle='->',linewidth=0.8,mutation_scale=7))
    ax1a.annotate('',xy=(2.9e6,3e7),xytext=(14e5,1e7),arrowprops=dict(color='tab:red',arrowstyle='->',linewidth=0.8,mutation_scale=7))
    ax1a.text(x=5.8e6,y=2.5e7,s=(r'Increasing $\dot{M},L$'),fontsize=7,color='tab:blue',ha='left',va='center',rotation=-32)
    ax1a.text(x=15e5,y=5e6,s=(r'Increasing $r_\mathrm{ph},L$'),fontsize=7,color='tab:red',ha='left',va='center',rotation=-16)

    # ax1b.annotate('',xy=(7e6,1e-2),xytext=(w0.r[iw],w0.rho[iw]),arrowprops=dict(color='tab:blue',arrowstyle='->',linewidth=0.8,mutation_scale=7))
    ax1b.annotate('',xy=(5.5e6,5e-3),xytext=(3e6,7e-5),arrowprops=dict(color='tab:blue',arrowstyle='->',linewidth=0.8,mutation_scale=7))
    ax1b.annotate('',xy=(9e6,1e-7),xytext=(14e5,1e-7),arrowprops=dict(color='tab:red',arrowstyle='->',linewidth=0.8,mutation_scale=7))

    ax1c.annotate('',xy=(2.7e6,3e5),xytext=(12.5e5,1e6),arrowprops=dict(color='tab:blue',arrowstyle='->',linewidth=0.8,mutation_scale=7))

    ax1d.annotate('',xy=(7.5e6,1.5e-3),xytext=(3e6,3e-2),arrowprops=dict(color='tab:blue',arrowstyle='->',linewidth=0.8,mutation_scale=7))
    ax1d.annotate('',xy=(2e6,1e-2),xytext=(11.8e5,1e-1),arrowprops=dict(color='tab:red',arrowstyle='->',linewidth=0.8,mutation_scale=7))


    ax1d.annotate('',xy=(1e8,1e-2),xytext=(1e8,5e-4),arrowprops=dict(color='k',arrowstyle='->',linewidth=0.8,mutation_scale=7))
    ax1d.annotate('',xy=(1e8,1e-6),xytext=(1e8,2e-5),arrowprops=dict(color='k',arrowstyle='->',linewidth=0.8,mutation_scale=7))
    ax1d.text(x=1.2e8,y=8e-4,s='optically\n thin',ha='left',va='bottom',fontsize=7)
    ax1d.text(x=1.2e8,y=1.5e-5,s='optically\n thick',ha='left',va='top',fontsize=7)
    ax1d.text(x=1.5e6,y=1.05,s='streaming limit',ha='left',va='bottom',fontsize=7)

    box,trans = (0.65,0.97),fig.transFigure # works for mnras1col
    ax1a.legend(frameon=False,ncol=2,bbox_to_anchor=box, bbox_transform=trans,fontsize=10)

    ## set y ticks
    for ax in (ax1b,ax1d):
        y_major = LogLocator(base = 10.0, numticks = 5)
        ax.yaxis.set_major_locator(y_major)
        y_minor = LogLocator(base = 10.0,numticks=20)
        ax.yaxis.set_minor_locator(y_minor)
        ax.yaxis.set_minor_formatter(NullFormatter())

    # plt.show()

    fig.savefig('paper_plots/profiles.'+img,bbox_inches='tight',format=img,dpi=dpi)
    print('Saved figure to paper_plots/profiles.'+img)


def Make_gradients_plot(figsize):

    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(figsize[0],3*figsize[1]),sharex=True)

    fig.subplots_adjust(hspace=0,wspace=0.25)

    ax3.set_xlabel(r'r (cm)')

    ax1.set_ylabel(r'$d\ln T/d\ln r$')
    ax2.set_ylabel(r'$d\ln\rho/d\ln r$')
    ax3.set_ylabel(r'$d\ln v/d\ln r$')

    for ax in (ax1,ax2,ax3):
        ax.set_xlim([1e6,1e8])

    # Winds
    for i,logMdot in enumerate([17.25,17.5,17.75,18,18.25,18.5]):

        w = IO.load_wind(logMdot)
        rphot,rsonic = w.rph,w.rs

        flnT,flnrho,flnv = IUS(np.log(w.r),np.log(w.T)), IUS(np.log(w.r),np.log(w.rho)), IUS(np.log(w.r),np.log(w.u))
        rvec = np.logspace(np.log10(w.r[1]),np.log10(w.r[-1]),1000)

        isonic = np.argmin(abs(rvec-rsonic))
        iphot = np.argmin(abs(rvec-rphot))

        # Discrete derivative
        x = np.log(rvec)
        y1,y2,y3 = flnT(x),flnrho(x),flnv(x)

        # dy1_dx = (y1[1:]-y1[:-1])/(x[1:]-x[:-1])
        # dy2_dx = (y2[1:]-y2[:-1])/(x[1:]-x[:-1])
        # dy3_dx = (y3[1:]-y3[:-1])/(x[1:]-x[:-1])
        # x=x[1:]

        dy1_dx = (y1[2:]-y1[:-2])/(x[2:]-x[:-2])
        dy2_dx = (y2[2:]-y2[:-2])/(x[2:]-x[:-2])
        dy3_dx = (y3[2:]-y3[:-2])/(x[2:]-x[:-2])
        x=x[1:-1]

        for dy_dx,ax in zip((dy1_dx,dy2_dx,dy3_dx), (ax1,ax2,ax3)):
            ax.semilogx(np.exp(x),dy_dx,ls='-',color='tab:blue',lw=0.5)
            ax.semilogx(np.exp(x)[isonic],dy_dx[isonic],marker='x',color='tab:blue',ms=2)
            ax.semilogx(np.exp(x)[iphot],dy_dx[iphot],marker='.',color='tab:blue',ms=3)


        # For arrows
        if logMdot==17.25:
            flnTi,flnrhoi,flnvi = flnT,flnrho,flnv
        elif logMdot==18.5:
            flnTf,flnrhof,flnvf = flnT,flnrho,flnv

    # Envelopes
    Rphotkms = (13,15,20,30,40,50)
    for i,Rphotkm in enumerate(Rphotkms):

        env = IO.load_envelope(Rphotkm)
        rphot = env.rph

        r,T,rho = env.r[env.rho>1e-20],env.T[env.rho>1e-20],env.rho[env.rho>1e-20]

        flnT,flnrho = IUS(np.log(r),np.log(T)), IUS(np.log(r),np.log(rho))
        rvec = np.exp(np.linspace(np.log(r[0]), np.log(r[-1]),100))
        iphot = np.argmin(abs(rvec-rphot))

        # Discrete derivative
        x = np.log(rvec)
        y1,y2 = flnT(x),flnrho(x)

        # dy1_dx = (y1[1:]-y1[:-1])/(x[1:]-x[:-1])
        # dy2_dx = (y2[1:]-y2[:-1])/(x[1:]-x[:-1])
        # x=x[1:]

        dy1_dx = (y1[2:]-y1[:-2])/(x[2:]-x[:-2])
        dy2_dx = (y2[2:]-y2[:-2])/(x[2:]-x[:-2])
        x=x[1:-1]

        # dlnT/dlnr is supposed to go to -0.5 but we cut that part when we remove rho<1e-20
        x2=x # no change for rho
        x1=np.append(x,[x[-1]*1.001,np.log(1e8)])
        dy1_dx=np.append(dy1_dx,[-0.5,-0.5])

        for x,dy_dx,ax in zip((x1,x2),(dy1_dx,dy2_dx),(ax1,ax2)):
            ax.semilogx(np.exp(x),dy_dx,ls='-',color='tab:red',lw=0.5)
            ax.semilogx(np.exp(x[iphot]),dy_dx[iphot],marker='.',color='tab:red',ms=3)


    ax1.set_ylim([-4.3,0])
    ax2.set_ylim([-9.3,-1.7])
    ax3.set_ylim([0,4.5])

    # Arrows
    ax1.annotate('',xy=(2.2e6,-2),xytext=(1.5e6,-1.65),arrowprops=dict(color='tab:blue',arrowstyle='->',linewidth=0.8,mutation_scale=7))
    ax2.annotate('',xy=(2.5e6,-6.5),xytext=(1.6e6,-5.8),arrowprops=dict(color='tab:blue',arrowstyle='->',linewidth=0.8,mutation_scale=7))
    ax3.annotate('',xy=(6e6,2.6),xytext=(2.4e6,2),arrowprops=dict(color='tab:blue',arrowstyle='->',linewidth=0.8,mutation_scale=7))

    fig.savefig('paper_plots/gradients.'+img,bbox_inches='tight',format=img,dpi=dpi)
    print('Saved figure to paper_plots/gradients.'+img)


def Make_rho_T_plot(figsize):

    fig,ax = plt.subplots(1,1,figsize=figsize)

    ax.set_xlabel(r'$\rho$ (g cm$^{-3}$)')
    ax.set_ylabel(r'$T$ (K)')

    ax.set_xlim([1e-8,1e6])
    ax.set_ylim([1e6,3e9])

    # Winds
    for i,logMdot in enumerate([17.25,17.5,17.75,18,18.25,18.5]):
        w = IO.load_wind(logMdot)

        iphot = list(w.r).index(w.rph)
        isonic = list(w.r).index(w.rs)

        label = 'winds' if i==0 else None

        ax.loglog(w.rho,w.T,color='tab:blue',ls='-',lw=0.5,label=label)
        ax.loglog(w.rho[isonic],w.T[isonic],color='tab:blue',marker='x',ms=2)
        ax.loglog(w.rho[iphot],w.T[iphot],color='tab:blue',marker='.',ms=3)

        if logMdot==17.25:
            ax.loglog(w.rho,w.T,color='tab:blue',ls='-',lw=0.5,label=label)

    # Envelopes
    Rphotkms = (13,15,20,30,40,50)
    for i,Rphotkm in enumerate(Rphotkms):
        env = IO.load_envelope(Rphotkm)
        iphot = list(env.r).index(env.rph)

        label = 'envelopes' if i==0 else None

        ax.loglog(env.rho,env.T,color='tab:red',ls='-',lw=0.5,label=label)
        ax.loglog([env.rho[iphot]],[env.T[iphot]],color='tab:red',marker='.',ms=3)

        if Rphotkm==13:
            ax.loglog(env.rho,env.T,color='tab:red',ls='-',lw=0.5,label=label)


    ## Pressure lines
    # Helium EOS
    X,Z,mu_I = 0,2,4
    mu_e = 2/(1+ X)
    mu = 1/(1/mu_I + 1/mu_e)

    Rho = np.logspace(-6,10,100)
    Knr,Kr = 9.91e12/mu_e**(5/3), 1.231e15/mu_e**(4/3)

    # Prad = Pg (non-degen)
    T1 = (3*kB*Rho/(arad*mu*mp))**(1/3)

    # Pednr = Pend (non-degen) : Knr rho**(5/3) = kTrho/mu_e*mp
    T2 = Knr*mu_e*mp/kB * Rho**(2/3)

    # Pedr = Pednr
    rho_rel = (Kr/Knr)**3

    ax.loglog(Rho,T1,'k-',lw=0.3)
    ax.loglog(Rho,T2,'k-',lw=0.3)
    ax.axvline(rho_rel,color='k',lw=0.7)


    ax.text(Rho[np.argmin(np.abs(T1-3e6))]*2,3e6,(r'$P_R=P_g$'),
        transform=ax.transData,ha='left',va='center',fontsize=8)

    ax.text(Rho[np.argmin(np.abs(T2-3e6))]*2,3e6,(r'$P_\mathrm{nd}=P_\mathrm{d}$'),
        transform=ax.transData,ha='left',va='center',fontsize=8)


    # Arrows
    ax.annotate('',xy=(5e1,7.5e8),xytext=(1e0,1.4e8),arrowprops=dict(color='tab:blue',arrowstyle='->',linewidth=0.8,mutation_scale=7))
    ax.annotate('',xy=(1e-7,1.7e6),xytext=(4e-8,3e6),arrowprops=dict(color='tab:blue',arrowstyle='->',linewidth=0.8,mutation_scale=7))

    ax.annotate('',xy=(1e-1,1e8),xytext=(1e-2,3.6e7),arrowprops=dict(color='tab:red',arrowstyle='->',linewidth=0.8,mutation_scale=7))
    ax.annotate('',xy=(1e-7,5.5e6),xytext=(1e-7,1.9e7),arrowprops=dict(color='tab:red',arrowstyle='->',linewidth=0.8,mutation_scale=7))


    ## set x ticks
    x_major = LogLocator(base = 10.0, numticks = 5)
    ax.xaxis.set_major_locator(x_major)
    x_minor = LogLocator(base = 10.0,numticks=20)
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(NullFormatter())

    # plt.show()

    fig.savefig('paper_plots/rho_T.'+img,bbox_inches='tight',format=img,dpi=dpi)
    print('Saved figure to paper_plots/rho_T.'+img)


def Make_luminosity_plot(figsize,ion=False):

    fig,ax1 = plt.subplots(1,1,figsize=figsize)

    ax1.set_xlabel(r'$\rho$ (g cm$^{-3}$)')
    ax1.set_xlim([1e-9,1e2])
    ax1.set_ylabel(r'$L/L_\mathrm{cr}$')

    ax1.set_ylim([0.995,1.01])
    ax1.axhline(1,color='k',ls='--',lw=0.7)

    # inset plot
    ax2 = fig.add_axes([0.5, 0.53, 0.34, 0.27])
    ax2.set_xlim([1e-6,1e-3])
    ax2.set_ylim([0.9998,1])

    markersizes = ((0,1),(1.8,0.5)) # sonic and phot marker sizes for main plot and inset

    M,comp = IO.load_params()['M'], IO.load_params()['comp']
    gr,eos = physics.GeneralRelativity(M), physics.EquationOfState(comp)

    # Winds
    for i,logMdot in enumerate([17.25,17.5,17.75,18,18.25,18.5]):

        w = IO.load_wind(logMdot)

        rphot = w.rph
        iphot = list(w.r).index(rphot)
        isonic = list(w.r).index(w.rs)

        label = 'winds' if i==0 else None

        x = w.rho

        Lratio = gr.Lcomoving(w.Lstar,w.r,w.u)/gr.Lcrit(w.r,w.rho,w.T,eos)
        y=Lratio

        for ax,ms in zip((ax1,ax2),markersizes):
            ax.semilogx(x,y,ls='-',color='tab:blue',lw=0.3,label=label)
            ax.semilogx(x[isonic],y[isonic],marker='x',color='tab:blue',ms=ms[0],markerfacecolor='b')
            ax.semilogx(x[iphot],y[iphot],marker='o',color='tab:blue',ms=ms[1],markerfacecolor='b')

            if logMdot==17.25:
                ax.semilogx(x,y,ls='-',color='tab:blue',lw=0.3,label=label)

    # Envelopes
    Rphotkms = (13,15,20,30,40,50)
    for i,Rphotkm in enumerate(Rphotkms):

        env = IO.load_envelope(Rphotkm)
        iphot = list(env.r).index(env.rph)

        label = 'envelopes' if i==0 else None

        x = env.rho

        Lratio = env.Linf*gr.Swz(env.r)**(-1)/gr.Lcrit(env.r,env.rho,env.T,eos)
        y = Lratio

        for ax,ms in zip((ax1,ax2),markersizes):
            ax.semilogx(x,y,ls='-',color='tab:red',lw=0.3,label=label)
            ax.semilogx(x[iphot],y[iphot],marker='o',color='tab:red',ms=ms[1],markerfacecolor='r')

            if Rphotkm==13:
                ax.semilogx(x,y,ls='-',color='tab:red',lw=0.3,label=label)


    # Arrows
    ax1.annotate('',xy=(2e1,0.997),xytext=(6e-2,0.997),arrowprops=dict(color='tab:blue',arrowstyle='->',linewidth=0.6,mutation_scale=6))
    ax1.annotate('',xy=(1e-8,1.0069),xytext=(1e-8,1.0007),arrowprops=dict(color='tab:blue',arrowstyle='->',linewidth=0.6,mutation_scale=6))

    ax1.annotate('',xy=(1.1e-2,0.997),xytext=(5e-4,0.997),arrowprops=dict(color='tab:red',arrowstyle='->',linewidth=0.6,mutation_scale=6))
    ax1.annotate('',xy=(1e-8,0.9993),xytext=(1e-8,0.9968),arrowprops=dict(color='tab:red',arrowstyle='->',linewidth=0.6,mutation_scale=6))



    # TICKS and stuff..

    # set x ticks
    x_major = LogLocator(base = 10.0, numticks = 6)
    ax1.xaxis.set_major_locator(x_major)
    x_minor = LogLocator(base = 10.0,numticks=20)
    ax1.xaxis.set_minor_locator(x_minor)
    ax1.xaxis.set_minor_formatter(NullFormatter())

    ax2.set_xticks([1e-5,1e-4])

    # set yticks
    y_major = FixedLocator([0.995,1,1.005,1.01])
    ax1.yaxis.set_major_locator(y_major)
    y_minor = FixedLocator([0.9975,1.0025,1.0075])
    ax1.yaxis.set_minor_locator(y_minor)
    ax1.yaxis.set_minor_formatter(NullFormatter())

    y_major = FixedLocator([0.9998,1])
    ax2.yaxis.set_major_locator(y_major)
    y_minor = FixedLocator([0.9999])
    ax2.yaxis.set_minor_locator(y_minor)
    ax2.yaxis.set_minor_formatter(NullFormatter())

    ax2.tick_params(axis='both', which='major', labelsize=7)

    for side in ['top','bottom','left','right']:
        ax2.spines[side].set_linewidth(0.5)

    ax2.xaxis.set_tick_params(width=0.5)
    ax2.yaxis.set_tick_params(width=0.5)

    # plt.show()

    fig.savefig('paper_plots/luminosity.'+img,bbox_inches='tight',format=img,dpi=dpi)
    print('Saved figure to paper_plots/luminosity.'+img)


def Make_opticaldepth_plot(figsize):

    fig,ax = plt.subplots(1,1,figsize=figsize)

    ax.set_xlabel(r'r (cm)')
    ax.set_ylabel(r'$\tau,\tau^*$')

    ax.set_ylim([1e-1,100])
    ax.set_xlim([1e6,1e9])

    def taustar(r,rho,T):
        return rho*eos.kappa(rho,T)*r

    def tau_true(r,rho,T, npts, min_rho=1e-20):

        # We want tau(r), i.e. we want a value at many radii point. We can do this by splitting the interval
        # in npts, and quad-integrate a spline interpolation between every point, and then yield the
        # cumulative sums

        # Cut any points that have rho=min_rho, because presumably min_rho was just assigned, is not physical
        if min_rho in rho:
            r,rho,T = r[rho>min_rho],rho[rho>min_rho],T[rho>min_rho]

        if len(r)<npts:
            raise Exception('npts (%d) larger than number of data points (%d)'%(npts,len(r)))

        frho,fT = IUS(r,rho),IUS(r,T)
        def dtau(ri):
            return eos.kappa(frho(ri),fT(ri))*frho(ri)/gr.Swz(ri)**0.5

        rvec = np.logspace(np.log10(r[0]),np.log10(r[-1]),npts)

        partial_tau = []
        for i in range(npts-1): # that is the number of intervals
            partial_tau.append(quad(dtau,rvec[i],rvec[i+1],limit=200,epsrel=1e-5)[0])

        # Calculate tau by taking the cumulative sum, in reverse (starting at largest r)
        tau = [0]
        for t in partial_tau[::-1]:
            tau.append(tau[-1] + t)

        tau = tau[::-1] # reverse back
        return rvec,partial_tau,tau

    for i,Rphotkm in enumerate((15,30,60)):
        env = IO.load_envelope(Rphotkm)
        rvec,partial_tau,tau = tau_true(env.r,env.rho,env.T,npts=500)

        label=(r'$\tau$') if i==0 else None
        ax.loglog(rvec,tau,color='tab:red',ls='-',lw=0.5)

        iphot = np.argmin(abs(rvec-env.rph))
        ax.loglog(rvec[iphot],tau[iphot],color='tab:red',marker='.',ms=3)

        label=(r'$\tau^*$') if i==0 else None
        ax.loglog(env.r,taustar(env.r,env.rho,env.T),color='tab:red',ls='--',lw=0.5)

    for i,logMdot in enumerate((17,18,18.5)):

        w = IO.load_wind(logMdot)
        rvec,partial_tau,tau = tau_true(w.r,w.rho,w.T,npts=1000)

        label=(r'$\tau$') if i==0 else None
        ax.loglog(rvec,tau,color='tab:blue',ls='-',lw=0.5)

        rphot = w.rph
        iphot = np.argmin(abs(rvec-rphot))
        isonic = np.argmin(abs(rvec-w.rs))

        ax.loglog(rvec[isonic],tau[isonic],color='tab:blue',marker='x',ms=2)
        ax.loglog(rvec[iphot],tau[iphot],color='tab:blue',marker='.',ms=3)
        # iphot2=list(w.r).index(rphot)
        # print(logMdot,tau[iphot],taustar(w.r[iphot2],w.rho[iphot2],w.T[iphot2]))

        ax.loglog(w.r,taustar(w.r,w.rho,w.T),color='tab:blue',ls='--',lw=0.5)

    ax.axhline(2/3,color='k',ls=':',lw=0.5)
    ax.axhline(3,color='k',ls=':',lw=0.5)
    ax.text(5e8,3,'3',color='k',ha='left',va='bottom',fontsize=7)
    ax.text(5e8,2/3,'2/3',color='k',ha='left',va='bottom',fontsize=7)

    class MyObjectHandler(HandlerBase):
        def create_artists(self, legend, orig_handle,
                        x0, y0, width, height, fontsize, trans):

                l1 = plt.Line2D([x0,width], [0.6*height,0.6*height], linewidth=0.8, color=orig_handle[0], ls=orig_handle[2])
                l2 = plt.Line2D([x0,width], [0.2*height,0.2*height], linewidth=0.8, color=orig_handle[1], ls=orig_handle[2])

                return l1,l2

    ax.legend([('tab:blue','tab:red','-'), ('tab:blue','tab:red','--')], [(r'$\tau$'),(r'$\tau^*$')],
            handler_map={tuple: MyObjectHandler()}, frameon=False, loc=1)

    # plt.show()

    fig.savefig('paper_plots/optical_depth.'+img,bbox_inches='tight',format=img,dpi=dpi)
    print('Saved figure to paper_plots/optical_depth.'+img)


def Make_triple_plot(figsize):

    fig,(ax1,ax2,ax3) = plt.subplots(3,1, sharex=True, figsize=(figsize[0],3*figsize[1]))
    fig.subplots_adjust(hspace=0)

    ax3.set_xlabel(r'$L_b^\infty/L_\mathrm{Edd}$')
    ax1.set_ylabel(r'T$_\mathrm{b}$ (10$^{9}$ K) ')
    ax2.set_ylabel(r'$r$ (km)')

    for ax in (ax1,ax2,ax3):
        ax.set_xlim([0.8,2.8])

    # Timescale expressions
    def soundtime(r,T,a,b):
        cs = np.sqrt(eos.cs2(T))
        func_inverse_cs = interp1d(r,1/cs,kind='linear')
        tsound,_ = quad(func_inverse_cs,a,b,epsrel=1e-5)
        return tsound

    def soundtime2(r,T,rs):
        isonic = list(r).index(rs)
        cs_at_rs = np.sqrt(eos.cs2(T[isonic]))
        return rs/cs_at_rs

    def soundflowtime(r,T,u,a,b,c):
        cs = np.sqrt(eos.cs2(T))
        func_inverse_cs = interp1d(r,1/cs,kind='linear')
        func_inverse_u = interp1d(r,1/u,kind='linear')
        t1,_ = quad(func_inverse_cs,a,b,epsrel=1e-5)
        t2,_ = quad(func_inverse_u,b,c,epsrel=1e-5)
        return t1+t2


    ## Load grid data
    logMdots = IO.get_wind_list()
    Lbinf_wind,Tb_wind,rph_wind,rs_wind = [[] for i in range(4)]
    tsound_wind,tsound2_wind,tsoundflow_wind = [[] for i in range(3)]
    for logMdot in [x for x in logMdots if x<=18.5]:
        w = IO.load_wind(logMdot)
        Lbinf_wind.append(w.Lstar[0])
        Tb_wind.append(w.T[0])
        rph_wind.append(w.rph)
        rs_wind.append(w.rs)

        tsound_wind.append(soundtime(w.r,w.T,w.r[0],w.rph))
        tsound2_wind.append(soundtime2(w.r,w.T,w.rs))
        tsoundflow_wind.append(soundflowtime(w.r,w.T,w.u,w.r[0],w.rs,w.rph))

    Lbinf_wind,Tb_wind,rph_wind,rs_wind = np.array((Lbinf_wind,Tb_wind,rph_wind,rs_wind))

    rph_env = [x for x in IO.get_envelope_list() if x>=12.1]
    Linf_env,Tb_env,tsound_env = [[] for i in range(3)]
    for rph in rph_env:
        env = IO.load_envelope(rph)
        Linf_env.append(env.Linf)
        Tb_env.append(env.T[0])

        tsound_env.append(soundtime(env.r,env.T,env.r[0],env.rph))

    Linf_env,Tb_env = np.array((Linf_env,Tb_env))

    Trad = (3*Pb/arad)**(1/4)

    ax1.axhline(Trad/1e9, color='k', linestyle='--', lw=0.4)
    ax1.text(Lbinf_wind[0]/LEdd, Trad/1e9-0.03, 'Radiation Pressure Limit',fontsize=7)
    ax1.axvline(1, color='k', linestyle='--', lw=0.4)
    ax2.axvline(1, color='k', linestyle='--', lw=0.4)
    ax3.axvline(1, color='k', linestyle='--', lw=0.4)

    ax1.set_ylim([1.17,1.62])

    ax1.plot(Linf_env/LEdd, Tb_env/1e9, color='tab:red',ls='-', lw=0.8, label='envelopes')
    ax1.plot(Lbinf_wind/LEdd, Tb_wind/1e9, color='tab:blue',ls='-', lw=0.8, label='winds')

    box = (0.8,1.15)
    ax1.legend(frameon=False, ncol=2, bbox_to_anchor=box, bbox_transform=ax1.transAxes)


    # Mdots labeling
    fulltext_pos = 18
    write_number = [17,17.7,18,18.2,18.3,18.4,18.5]
    write_number.remove(fulltext_pos)
    x = Lbinf_wind/LEdd
    for i in range(len(logMdots)):
        if logMdots[i] == fulltext_pos:
                ax1.text(x[i]+0.05, Tb_wind[i]/1e9-0.02, (r'log $\dot{M}$ = '+str(logMdots[i])),
                        fontweight='bold', color='k',fontsize=7)
                ax1.plot(x[i], Tb_wind[i]/1e9, 'ko', mfc='w', mec='k', ms=3, mew=.5)
        elif logMdots[i] in write_number:
                ax1.text(x[i]+0.05, Tb_wind[i]/1e9-0.02, str(logMdots[i]),
                        fontweight='bold', color='k',fontsize=7)
                ax1.plot(x[i], Tb_wind[i]/1e9, 'ko', mfc='w', mec='k', ms=3, mew=.5)


    ## set x ticks
    x_major = FixedLocator([1,1.5,2,2.5])
    ax.xaxis.set_major_locator(x_major)
    x_minor = FixedLocator([1.25,1.75,2.25,2.75])
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(NullFormatter())


    ax2.semilogy(Lbinf_wind/LEdd, rph_wind/1e5, color='tab:blue',ls='-',lw=0.8)#,label=r'$r_\mathrm{ph}$')
    ax2.semilogy(Linf_env/LEdd, rph_env, color='tab:red',ls='-',lw=0.8)
    ax2.semilogy(Lbinf_wind/LEdd, rs_wind/1e5, color='tab:blue',ls='-.',lw=0.8)#, label=r'$r_s$')
    ax2.set_ylim([7,2e3])

    ## Timescales in seconds
    l1,=ax3.semilogy(Linf_env/LEdd, tsound_env, color='tab:red',ls='-', lw=0.8)#, label=r'$\int c_s^{-1}dr$')
    l2,=ax3.semilogy(Lbinf_wind/LEdd, tsound_wind, color='tab:blue',ls='-', lw=0.8)#, label=r'$\int c_s^{-1}dr$')
    l3,=ax3.semilogy(Lbinf_wind/LEdd, tsound2_wind, color='tab:blue',ls=':', lw=0.8)# label=r'$r_s/c_s(r_s)$')
    l4,=ax3.semilogy(Lbinf_wind/LEdd, tsoundflow_wind, color='tab:blue',ls='-.', lw=0.8)#, label=r'$\int_{R}^{r_c} c_s^{-1}dr+\int_{r_c}^{r_{ph}} u^{-1}dr$')
    ax3.set_ylabel('Time (s)')
    ax3.set_ylim([6e-4,2e1])


    # Inspired by https://stackoverflow.com/questions/41752309/single-legend-item-with-two-lines
    class MyObjectHandler(HandlerBase):
        def create_artists(self, legend, orig_handle,
                        x0, y0, width, height, fontsize, trans):

                if not set(orig_handle[1]).isdisjoint(('b','k','g','r')):  # could add more colors to this

                        l1 = plt.Line2D([x0,width], [0.7*height,0.7*height], linewidth=0.8, color=orig_handle[0])
                        l2 = plt.Line2D([x0,width], [0.3*height,0.3*height], linewidth=0.8, color=orig_handle[1])

                        return l1, l2

                else:
                        l1 = plt.Line2D([x0,y0+width], [0.5*height,0.5*height], linewidth=0.8,
                                        color=orig_handle[0], linestyle=orig_handle[1])

                        l2 = plt.Line2D([x0,width], [0.5*height,0.5*height], alpha=0)  #transparent

                        return l1,l2

    ax3.legend([('tab:blue','tab:red'), ('tab:blue','-.'), ('tab:blue',':')], [(r'$\tau_\mathrm{sound}$'),(r'$\tau_\mathrm{sound-flow}$'),(r'$\tau_s$')],
           handler_map={tuple: MyObjectHandler()}, frameon=False, loc=4)

    ax2.legend([('tab:blue','tab:red'), ('tab:blue','-.')], [(r'$r_\mathrm{ph}$'),(r'$r_s$')], handler_map={tuple: MyObjectHandler()}, frameon=False, bbox_to_anchor=(0.9,0.25), bbox_transform=ax2.transAxes)


    # plt.show()

    fig.savefig('paper_plots/triple.'+img, format=img,dpi=dpi, bbox_inches='tight')
    print('Saved figure to paper_plots/triple.'+img)


def Make_spot_check_MR_plot(figsize):

    fig,ax= plt.subplots(1,1, figsize=(0.7*figsize[0],0.7*figsize[1]))
    ax.set_xlabel(r'$L_b^\infty/L_\mathrm{Edd}$')
    ax.set_xlim([0.7,2.8])

    ax.set_ylabel(r'$r_\mathrm{ph}$ (km)')

    for M,col in zip((1.1,1.4,2.0),('m','k','g')):

        LEdd = 4*np.pi*6.6726e-8*2e33*M*2.99292458e10/0.2

        for R,ls in zip((10,12,13),('--','-',':')):

            # if M==1.4 and R==12:
            #     wind_model_name = 'He_M' + str(M) + '_R' + str(R) + '_y8_exact'
            # else:
            #     wind_model_name = 'He_M' + str(M) + '_R' + str(R) + '_y8'

            # For the plot in the paper, we use the non-exact wind models (with Prad=aT^4/3 everywhere), because
            # they are more filled out across the Mdot range (easier to compute) and are very similar anyway.

            # wind_model_name = 'He_M' + str(M) + '_R' + str(R) + '_y8_exact/'
            wind_model_name = 'He_M' + str(M) + '_R' + str(R) + '_y8/'

            env_model_name  = 'He_M' + str(M) + '_R' + str(R) + '_y8/'

            label=('(%.1f;%d)'%(M,R))

            try:

                path = 'winds/models/' + wind_model_name

                logMdots = []
                for filename in os.listdir(path):
                    if filename.endswith('.txt'):
                        logMdots.append(eval(filename[:-4]))

                logMdots = list(np.sort(logMdots))

                Lbinf,Rphot = [],[]
                for logMdot in logMdots:
                    filename = path + ('%.2f'%logMdot) + '.txt'
                    w = IO.load_wind(logMdot,specific_file=filename)
                    Lbinf.append(w.Lstar[0])
                    Rphot.append(w.rph)

                x,y = [],[]
                for i in range(len(logMdots)):
                    if Lbinf[i]/LEdd <= 2.8:
                        x.append(Lbinf[i]/LEdd)
                        y.append(Rphot[i]/1e5)

                ax.semilogy(x, y, color=col, ls=ls, lw=0.9)

            except:
                print('winds do not exist for M=%s, R=%s'%(str(M),str(R)))

            try:
                path = 'envelopes/models/' + env_model_name

                Rphotkm = []
                for filename in os.listdir(path):
                    if filename.endswith('.txt'):
                        Rphotkm.append(eval(filename[:-4].replace('_','.')))

                Rphotkm = [int(x) if str(x)[-1]=='0' else x for x in list(np.sort(Rphotkm))]

                Rphotkm = [x for x in Rphotkm if x>=12.1] # don't show most compact models

                Linf = []
                for r in Rphotkm:
                    s = str(r)
                    if '.' in s:
                        if len(s[s.find('.')+1:]) == 1: # this counts the number of char after '.' (#decimals)
                            if s[-1]=='0':
                                r = round(eval(s))

                    filename = path + str(r).replace('.','_') + '.txt'
                    env = IO.load_envelope(r,specific_file=filename)
                    Linf.append(env.Linf)

                x = []
                for i in range(len(Rphotkm)):
                        x.append(Linf[i]/LEdd)

                ax.semilogy(x, Rphotkm, color=col, ls=ls, lw=0.9, label=label)

            except:
                print('envelopes do not exist for M=%s, R=%s'%(str(M),str(R)))

    ax.legend(ncol=3,title=r'($M/M_\odot$ ; $R/$km)')
    ax.set_yticks((1e2,1e3))

    # plt.show()

    fig.savefig('paper_plots/spot_check_MR.'+img, format=img,dpi=dpi, bbox_inches='tight')
    print('Saved figure to paper_plots/spot_check_MR.'+img)



if __name__ == '__main__':

    if not os.path.exists('paper_plots/'):
        os.mkdir('paper_plots/')

    onecol = set_size('mnras1col')
    twocol = set_size('mnras2col')


    # # wind plots
    # Make_lineshift_plot(twocol)
    # Make_wind_paramspace_plot(twocol)
    # Make_base_Enuc_plot(twocol)
    # Make_Mdot_prescription_plot(twocol)

    # # envelope plots
    # Make_touchdown_error_plot(twocol)
    # Make_env_paramspace_plot(twocol)
    # Make_env_energy_plot(twocol)

    # # compare plots
    # Make_profiles_plot(onecol)
    Make_gradients_plot(twocol)
    # Make_rho_T_plot(twocol)
    # Make_luminosity_plot(twocol)
    # Make_opticaldepth_plot(twocol)
    # Make_triple_plot(twocol)
    # Make_spot_check_MR_plot(onecol)


