''' Input and Output '''

import os
import numpy as np

#------------------------------------ Core ------------------------------------

def load_params(as_dict=True):
    ''' Load parameters given in params.txt '''

    with open('./params.txt','r') as f:
        M = float(f.readline().split()[1])
        R = float(f.readline().split()[1])
        yb = float(f.readline().split()[1])
        comp = f.readline().split()[1]
        Prad = f.readline().split()[1]
        
    if as_dict is True:
        return {'M':M,'R':R,'yb':yb,'comp':comp,'Prad':Prad}
    else:
        return M,R,yb,comp,Prad


def get_name(include_Prad = True):  
    ''' We give various files and directories the same name corresponding 
        to the setup given in the parameter file '''
    params = load_params()
    name = '_'.join([ 
        params['comp'], ('M%.1f'%params['M']), ('R%2d'%params['R']) , 
        ('y%1d'%np.log10(params['yb'])) ])

    if params['Prad'] == 'exact' and include_Prad:
        name += '_exact'
        
    return name


def get_wind_list():
    ''' Available winds for current model, listed by their logMdot (g/s) values '''
    
    path = 'winds/models/' + get_name() + '/'

    logMdots = []
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            logMdots.append(eval(filename[:-4]))

    sorted_list = list(np.sort(logMdots))
    return sorted_list


def get_envelope_list():
    ''' Available envelopes for current model, listed by their r_ph (km) values '''

    path = 'envelopes/models/' + get_name(include_Prad=False) + '/'

    Rphotkms = []
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            Rphotkms.append(eval(filename[:-4].replace('_','.')))

    sorted_list = list(np.sort(Rphotkms))
    sorted_list_clean = [int(x) if str(x)[-1]=='0' else x for x in sorted_list] # changes 15.0 to 15 for filename cleanliness
    return sorted_list_clean

    
#------------------------------------ Winds ------------------------------------

def save_wind_root(logMdot,root,decimals=8):

    filename = get_name()
    path = 'roots/roots_' + filename + '.txt'

    if not os.path.exists(path):
        f = open(path,'w+')
        f.write('{:<7s} \t {:<12s} \t {:<12s}\n'.format(
            'logMdot' , 'Edot/LEdd' , 'log10(Ts)'))
    else:
        f = open(path,'a')

    if decimals == 8:
        f.write('{:<7.2f} \t {:<11.8f} \t {:<11.8f}\n'.format(
                logMdot,root[0],root[1]))
    elif decimals == 9:
        f.write('{:<7.2f} \t {:<12.9f} \t {:<12.9f}\n'.format(
                logMdot,root[0],root[1]))
    elif decimals == 10:
        f.write('{:<7.2f} \t {:<13.10f} \t {:<13.10f}\n'.format(
                logMdot,root[0],root[1]))


def load_wind_roots(logMdot=None,specific_file=None):

    if specific_file != None:
        filename = specific_file
    else:
        name = get_name()
        path = 'winds/roots/roots_'
        filename = path + name + '.txt'

    if not os.path.exists(filename):
        raise TypeError('Root file does not exist (%s)'%filename)

    else:
        logMdots,roots = [],[]
        with open(filename, 'r') as f:
            next(f)
            for line in f:
                stuff = line.split()
                logMdots.append(float(stuff[0]))
                roots.append([float(stuff[1]), float(stuff[2])])

        if logMdot is None:
            return logMdots,roots
        else:
            return roots[logMdots.index(logMdot)]


def clean_wind_rootfile(warning=1):

    # Find duplicates, and remove all but the latest root 
    # (assuming the last one is the correct one)
    # Sort from lowest to biggest

    logMdots,roots = load_wind_roots() 
    new_logMdots = np.sort(np.unique(logMdots))

    if list(new_logMdots) != list(logMdots):

        v = []
        for x in new_logMdots:
            duplicates = np.argwhere(logMdots==x)
            v.append(duplicates[-1][0]) # keeping the last one

        new_roots = []
        for i in v:
            new_roots.append(roots[i])
            
        if warning:
            o = input('Roots file will be overwritten. Proceed? (0 or 1) ')
        else:
            o = 1
        if o:
            filename = get_name()
            path = 'roots/roots_' + filename + '.txt'
            os.remove(path)

            for x,y in zip(new_logMdots,new_roots): 
                decimals = max((len(str(y[0])),len(str(y[1])))) - 2   
                # print(decimals)             
                save_wind_root(x,y,decimals=decimals)


def write_wind(logMdot,wind):
    # Expecting wind type namedtuple object

    path = 'winds/models/' + get_name() + '/'
    filename = path + ('%.2f'%logMdot) + '.txt'
    # filename = path + ('%.2f'%logMdot) + 'TEMP.txt'

    with open(filename,'w') as f:

        # Write header
        f.write(('{:<11s} \t'*5 + '\t\t Edot= {:<6e}').format(
            'r (cm)','rho (g/cm3)','T (K)','u (cm/s)','L* (erg/s)',wind.Edot))

        f.write('\n')

        # Write values
        for i in range(len(wind.r)):
            f.write(('%0.6e \t'*5)%(
                wind.r[i], wind.rho[i], wind.T[i], wind.u[i], wind.Lstar[i]))

            if wind.r[i] == wind.rs:
                f.write('sonic point')

            if wind.r[i] == wind.rph:
                f.write('photospheric radius')

            f.write('\n')

    print('Wind data saved at: ',filename)


def load_wind(logMdot, specific_file=None):

    '''outputs arrays from save file and rs '''

    if specific_file != None:
        filename = specific_file
    else:
        path = 'winds/models/' + get_name() + '/'
        filename = path + ('%.2f'%logMdot) + '.txt'

    r,rho,T,u,Lstar = [[] for i in range(5)]
    varz = (r,rho,T,u,Lstar)
    with open(filename, 'r') as f:
        for i,line in enumerate(f):

            if i==0:
                Edot = float(line.split()[-1])
            
            else:
                append_vars(line, varz)

                if line.split()[-1] == 'point': 
                    rs = eval(line.split()[0])
                elif line.split()[-1] == 'radius':
                    rph = eval(line.split()[0])

    r,rho,T,u,Lstar = np.array(varz)


    # Return as wind tuple object
    from winds.wind_GR_FLD import Wind
    return Wind(rs, rph, Edot, r, T, rho, u, Lstar)


def save_EdotTsrel(logMdot, Edotvals, TsvalsA, TsvalsB, decimals=8):

    name = get_name()
    i = name.find('y')
    name = name[:i-1] + name[i+2:] # the inner column depth parameter is irrelevant for the EdotTsrelation

    path = 'winds/roots/rel/' + name

    if not os.path.exists(path):
        os.mkdir(path)

    filepath = path + '/EdotTsrel_' + ('%.2f'%logMdot) + '.txt'
    if not os.path.exists(filepath):
        f = open(filepath, 'w+')
        f.write('{:<12s} \t {:<12s} \t {:<12s}\n'.format(
                'Edot/LEdd', 'log10(TsA)', 'log10(TsB)'))

    else:
        f = open(filepath, 'a')

    for edot, tsa, tsb in zip(Edotvals, TsvalsA, TsvalsB):
        if decimals == 8:
            f.write('{:<11.8f} \t {:<11.8f} \t {:<11.8f}\n'.format(
                    edot, tsa, tsb))
        elif decimals == 10:
            f.write('{:<13.10f} \t {:<13.10f} \t {:<13.10f}\n'.format(
                    edot, tsa, tsb))


def load_EdotTsrel(logMdot, specific_file=None):

    if specific_file is not None:
        filepath = specific_file
    else:
        name = get_name()
        i = name.find('y')
        name = name[:i-1] + name[i+2:]

        filepath = 'winds/roots/rel/' + name + '/EdotTsrel_' + ('%.2f'%logMdot) + '.txt'
        if not os.path.exists(filepath):
            return False,

    Edotvals, TsvalsA, TsvalsB = [],[],[]
    with open(filepath,'r') as f:
        next(f)
        for line in f:
            Edotvals.append(eval(line.split()[0]))
            TsvalsA.append(eval(line.split()[1]))
            TsvalsB.append(eval(line.split()[2]))
    
    return True,Edotvals,TsvalsA,TsvalsB
    # note that Edotvals is (Edot-Mdotc^2)/LEdd, Tsvals is logTs


def clean_EdotTsrelfile(logMdot,warning=1):

    # Find duplicates, and remove all but the latest root 
    # (assuming the last one is the correct one)
    # Sort from lowest to biggest

    _,Edotvals,TsvalsA,TsvalsB = load_EdotTsrel(logMdot)
    new_Edotvals = np.sort(np.unique(Edotvals))

    # if list(new_Edotvals) != list(Edotvals):
    if True:

        v = []
        for x in new_Edotvals:
            duplicates = np.argwhere(Edotvals==x)
            v.append(duplicates[-1][0]) # keeping the last one

        new_TsvalsA, new_TsvalsB = [],[]
        for i in v:
            new_TsvalsA.append(TsvalsA[i])
            new_TsvalsB.append(TsvalsB[i])

        if warning:
            o = input('EdotTsrel file will be overwritten. Proceed? (0 or 1) ')
        else:
            o = 1
        if o:

            name = get_name()
            i = name.find('y')
            name = name[:i-1] + name[i+2:]

            filepath = 'winds/roots/rel/' + name + '/EdotTsrel_'+('%.2f'%logMdot)+'.txt'
            os.remove(filepath)

            # save_EdotTsrel(logMdot,new_Edotvals,new_TsvalsA,new_TsvalsB)
            for e,tsa,tsb in zip(new_Edotvals,new_TsvalsA,new_TsvalsB):
                decimals = max((len(str(e)),len(str(tsa)),len(str(tsb)))) - 2
                # print(decimals)
                save_EdotTsrel(logMdot,[e],[tsa],[tsb],decimals=decimals)
            


#---------------------------------- Envelopes ----------------------------------

def write_envelope(Rphotkm,env):
    # Expecting env type namedtuple object

    assert(env.rph/1e5==Rphotkm)

    path = 'envelopes/models/' + get_name(include_Prad=False) + '/'

    if Rphotkm >= load_params()['R']+1:
                filename = path + str(Rphotkm) + '.txt'
    else:
        filename = path + str(Rphotkm).replace('.','_') + '.txt'

    with open(filename,'w') as f:

        # Write header
        f.write('{:<13s} \t {:<11s} \t {:<11s} \t\t Linf = {:<6e}\n'.format(
            'r (cm)','rho (g/cm3)','T (K)',env.Linf))

        for i in range(len(env.r)):
            f.write('%0.8e \t %0.6e \t %0.6e \t'%
                (env.r[i], env.rho[i], env.T[i]))    

            f.write('\n')


def load_envelope(Rphotkm, specific_file=None):

    # output is Envelope namedtuple object       

    # Sometimes because of numpy coversions 13 gets converted to 13.0 for example.
    # We have to remove these zeros else the file isn't found
    s = str(Rphotkm)
    if '.' in s:
        if len(s[s.find('.')+1:]) == 1: # this counts the number of char after '.' (#decimals)
            if s[-1]=='0':
                Rphotkm = round(eval(s))

    if specific_file != None:
        filename = specific_file
    else:
        path = 'envelopes/models/' + get_name(include_Prad=False) + '/'

        if Rphotkm >= load_params()['R']+1:
                    filename = path + str(Rphotkm) + '.txt'
        else:
            filename = path + str(Rphotkm).replace('.','_') + '.txt'

    r, rho, T = [[] for i in range (3)]
    with open(filename,'r') as f:
        for i,line in enumerate(f): 
            if i==0: 
                Linf = float(line.split()[-1])
            else:
                append_vars(line,[r, rho, T])

    r,rho,T = np.array((r,rho,T))

    from envelopes.env_GR_FLD import Env
    return Env(Rphotkm*1e5,Linf,r,T,rho)


def save_rhophf0rel(Rphotkm, f0vals, rhophvalsA, rhophvalsB):

    path = 'envelopes/roots/' + get_name(include_Prad=False)

    if not os.path.exists(path):
        os.mkdir(path)

    filepath = path + '/rhophf0rel_' + str(Rphotkm) + '.txt'
    if not os.path.exists(filepath):
        f = open(filepath, 'w+')
        f.write('{:<12s} \t {:<12s} \t {:<12s}\n'.format(
                'f0', 'log10(rhophA)', 'log10(rhophB)'))
    else:
        f = open(filepath, 'a')

    for f0, rhopha, rhophb in zip(f0vals, rhophvalsA, rhophvalsB):
        f.write('{:<11.8f} \t {:<11.8f} \t {:<11.8f}\n'.format(
                f0, np.log10(rhopha), np.log10(rhophb)))


def load_rhophf0rel(Rphotkm):

    s = str(Rphotkm)
    if s[-2:]=='.0': s=s[:-2]

    filepath = 'envelopes/roots/' + get_name(include_Prad=False) + '/rhophf0rel_' + s + '.txt'
    if not os.path.exists(filepath):
        return False,

    else:
        f0, rhophA, rhophB = [],[],[]
        with open(filepath,'r') as f:
            next(f)
            for line in f:
                f0.append(eval(line.split()[0]))
                rhophA.append(10**eval(line.split()[1]))
                rhophB.append(10**eval(line.split()[2])) # saved as log values in the file
         
        return True,f0,rhophA,rhophB


def clean_rhophf0relfile(Rphotkm,warning=1):

    # Find duplicates, and remove all but the latest root 
    # (assuming the last one is the correct one)
    # Sort from lowest to biggest f0

    _,f0vals,rhophvalsA,rhophvalsB = load_rhophf0rel(Rphotkm)
    new_f0vals = np.sort(np.unique(f0vals))[::-1] # largest f0 value first (to work correctly in the initial search in MakeEnvelope)

    if list(new_f0vals) != list(f0vals):

        v = []
        for x in new_f0vals:
            duplicates = np.argwhere(f0vals==x)
            v.append(duplicates[-1][0]) # keeping the last one

        new_rhophvalsA, new_rhophvalsB = [],[]
        for i in v:
            new_rhophvalsA.append(rhophvalsA[i])
            new_rhophvalsB.append(rhophvalsB[i])

        if warning:
            o = input('EdotTsrel file will be overwritten. Proceed? (0 or 1) ')
        else:
            o = 1
        if o:
            filepath = 'envelopes/roots/'+get_name(include_Prad=False)+'/rhophf0rel_'+str(Rphotkm)+'.txt'
            os.remove(filepath)

            save_rhophf0rel(Rphotkm,new_f0vals,new_rhophvalsA,new_rhophvalsB)
    


#------------------------------------ Misc ------------------------------------

def append_vars(line,varz): 
    l=line.split()
    for col,var in enumerate(varz):
        var.append(float(l[col]))

def change_param(key, new_value):

    old_value = load_params()[key]

    if type(old_value) == str:
        old_value_string = old_value
        new_value_string = new_value

    else:

        if old_value>100: # we are looking at y_inner which is formatted as '1e8'
            old_value_string = '1e' + str(int(np.log10(old_value)))
            new_value_string = '1e' + str(int(np.log10(new_value)))

        elif old_value%1==0: # has no decimals
            old_value_string = str(int(old_value))
            new_value_string = str(int(new_value))

        else:
            old_value_string = str(old_value).rstrip('0') # remove trailing zeros if there are any
            new_value_string = str(new_value).rstrip('0') # remove trailing zeros if there are any


    new_file_contents = ""
    with open('./params.txt','r') as f:
        for line in f:
            if key in line:
                new_file_contents += line.replace(old_value_string,new_value_string)
            else:
                new_file_contents += line

    print(new_file_contents)

    with open('./params.txt','w') as f:
        f.write(new_file_contents)

def make_directories():
    ''' Make directories according to model name '''
    
    path1 = 'envelopes/models/' + get_name(include_Prad=False)
    path2 = 'winds/models/' + get_name()
    for path in (path1,path2):
        if not os.path.exists(path): 
            os.mkdir(path)


def export_grid(target = "."):

    ''' Export values of grid of models to target directory. Columns are Lbase (erg/s), Mdot, Edot (=Linf for envelopes)
        rs (=x for envelopes), rph, rhob, Tb, Teff '''

    if target[-1]!='/': target += '/'
    filename = target + 'grid_' + get_name(include_Prad=False)+'.txt'

    with open(filename, 'w') as f:

        # Header
        f.write(('{:<10s}    '*7 +'{:<10s}\n').format(
        'Lbinf (erg/s)','Mdot (g/s)','Edot (g/s)','rs (cm)','rph (cm)','rhob (g/cm3)','Tb (K)','Teff (K)'))

        # Envelopes
        for Rphotkm in get_envelope_list():
            env = load_envelope(Rphotkm)

            # Base luminosity
            f.write('%0.4e       ' % env.Linf)

            # Mdot,Edot
            f.write('0' + ' '*13 + '%0.4e    '%env.Linf)

            # rs,rph
            f.write('0' + ' '*13 + '%0.4e    ' % env.rph)

            # Base rho,T
            f.write('%0.4e      %0.4e    '%(env.rho[0],env.T[0]))

            # Teff (T(rph))
            f.write('%0.4e\n'%env.T[list(env.r).index(env.rph)])

        # Winds
        for logMdot in get_wind_list():
            w = load_wind(logMdot)

            # Base luminosity
            f.write('%0.4e       ' % w.Lstar[0])

            # Mdot,Edot
            f.write('%0.4e    %.4e    '%(10**logMdot,w.Edot))

            # rs,rph
            f.write('%0.4e    %.4e    '%(w.rs,w.rph))

            # Base rho,T
            f.write('%0.4e      %0.4e    '%(w.rho[0],w.T[0]))

            # Teff (T(rph))
            f.write('%0.4e\n'%w.T[list(w.r).index(w.rph)])

            


def load_wind_old(logMdot, specific_file=None):

    ''' Note to self : If wind text files are from old code, use this because the files are formatted different '''

    if specific_file != None:
        filename = specific_file
    else:
        path = 'winds/models/' + get_name() + '/'
        filename = path + ('%.2f'%logMdot) + '.txt'

    r,T,rho,P,u,cs,phi,L,Lstar,taus,lam = [[] for i in range(11)]
    varz = [r,T,rho,P,u,cs,phi,L,Lstar,taus,lam]
    with open(filename, 'r') as f:
        next(f)
        for line in f:
            append_vars(line, varz)

            if line.split()[-1] == 'point': 
                rs = eval(line.split()[0])

    r,T,rho,P,u,cs,phi,L,Lstar,taus,lam = (np.array(var) for var in varz)

    # Locate photosphere (should be in written down in text file like rs in the future)
    arad = 7.5657e-15
    c = 2.99792458e10
    F = L/(4*np.pi*r**2)
    x = F/(arad*c*T**4)
    rph = r[np.argmin(abs(x - 0.25))]

    # Calculate Edot
    import physics
    LEdd = 4*np.pi*c*6.6726e-8*2e33*load_params()['M'] / physics.EquationOfState(load_params()['comp'])
    Edot = load_wind_roots(logMdot)[0]*LEdd + 10**logMdot*c**2

    # Return as wind tuple object
    from winds.wind_GR_FLD import Wind
    return Wind(rs, rph, Edot, r, T, rho, u, Lstar)
