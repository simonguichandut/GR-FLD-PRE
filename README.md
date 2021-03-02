## Description

1D steady-state static envelopes and super-Eddington winds in photospheric radius expansion (PRE) Type I X-ray Bursts. Using GR and flux-limited diffusion. See Guichandut et al. (2021) for reference.

IO.py contains all the scripts for reading and writing. If you are not trying to make models, it's all you need.

    python3 IO.py
    import IO


# Load envelope models
IO.load_envelope(rph) loads the solution with rph (in km) into a namedTuple object (like a class). The radius, temperature, density, etc, are store and can be accessed with "."

    env = IO.load_envelope(50)
    plt.loglog(env.r,env.T)
    print(env.Linf)

# Load wind models
IO.load_wind(logMdot)

# Grid of models
Use IO.export_grid(dir), where dir is the target directory (default "./"). Produces a table with values for Lb, Mdot, Edot, Tb, rhob, etc.