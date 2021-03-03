1D steady-state static envelopes and super-Eddington winds in photospheric radius expansion (PRE) Type I X-ray Bursts. Using GR and flux-limited diffusion. See Guichandut et al. (2021) for reference.

Questions -> simon.guichandut@mail.mcgill.ca

Dependencies: python v>3.6, numpy, scipy.

# Existing models

IO.py contains all the scripts for reading and writing. If you are not trying to make new models, it's most likely all you need.

    python3     
    import IO   

## Load envelope models
`IO.load_envelope(rph)` loads the solution with rph (in km) into a 'Env' named tuple, which stores Linf (luminosity seen at infinity, one value), and r,T,rho (data points for radial coordinate, temperature, density). Example use:

    env = IO.load_envelope(50)
    plt.loglog(env.r,env.T)
    print(env.Linf)

## Load wind models
`IO.load_wind(logMdot)` loads the solution with logMdot into a 'Wind' namedTuple, which stores rs & rph (sonic point & photospheric radii), Edot (energy-loss rate), and r,T,rho,u,Lstar. u is gas velocity and Lstar is the luminosity seen at infinity with extra velocity factors (Lstar = LY^2(1+v^2/c^2), see Paczynski & Proszynski 1986). Example use:

    wind = IO.load_wind(18)
    plt.loglog(wind.r,wind.u)
    print(wind.Edot, wind.rs, wind.rph)

## Grid of models
Use `IO.export_grid(dir)`, where dir is the target directory (default "./"). Produces a table with values for Lbase at infinity, Mdot, Edot, rs, rph, rho_base, T_base, Teff. For envelopes, Mdot=0, Edot=Linf and rs=0.



# Making new models

To see the models that are already available, navigate to winds/models and envelopes/models and see the folder names. To make new models (e.g. different NS mass or radius), change the params.txt file.

For winds, you have to search the parameter space to find roots. This can work automatically to some extent with scripts in winds/Rootfinding.py, but it's probably best to contact me :).

Envelopes will likely work automatically, just run:

    python3 run.py envelope 15,20,50

Where the comma-separated values are rph values that you want solutions for.