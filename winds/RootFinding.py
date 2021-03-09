# !! Disclaimer. This file is just a bunch of scripts for navigating the complicated parameter spcae
# of these wind models, with a lot of tricks to speed up the computation time. Odds of understanding
# all the technicalities are slim, better to contact me directly :)

import sys
sys.path.append(".")

import numpy as np
import os
import IO
from scipy.interpolate import InterpolatedUnivariateSpline as IUS 
from scipy.optimize import fsolve,brentq
from winds.wind_GR_FLD import *

# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning) 

"""
Rootfinding is done differently in FLD than optically thick.  For every (Mdot,Edot), there is a single value of Ts 
that goes to infinity.  But it's not possible to find the exact value and shoot to infiniy.  Instead, for every Edot, 
we find two close (same to high precision) values of Ts that diverge in opposite direction, and declare either to be 
the correct one.  This gives a Edot-Ts relation, on which we can rootfind based on the inner boundary error.
"""

# M, RNS, y_inner, tau_out, comp, EOS_type, FLD, mode, save, img = IO.load_params(as_dict=False)

def run_outer(logMdot,Edot_LEdd,logTs,Verbose=0):  # full outer solution from sonic point
    global Mdot,Edot,Ts,verbose,rs
    Mdot, Edot, Ts, rs, verbose = setup_globals([Edot_LEdd,logTs],logMdot,Verbose=Verbose,return_them=True)
    
    if rs<IO.load_params()['R']*1e5: 
        raise Exception('sonic point less than RNS')

    # High rmax to ensure we find two solutions that diverge separatly in rootfinding process
    return outerIntegration(r0=rs, T0=Ts, phi0=2.0, rmax=1e11),rs

def run_inner(logMdot,Edot_LEdd,logTs,Verbose=0,solution=False):  # full inner solution from sonic point
    global Mdot,Edot,Ts,verbose,rs
    Mdot, Edot, Ts, rs, verbose = setup_globals([Edot_LEdd,logTs],logMdot,Verbose=Verbose,return_them=True)
    sol_inner1 = innerIntegration_r(rs,Ts)
    T95,phi95 = sol_inner1.sol(0.95*rs)
    _,rho95,_,_ = calculateVars_phi(0.95*rs, T95, phi=phi95, subsonic=True)

    if solution:
        return innerIntegration_rho(rho95, T95, returnResult=True)
    else:
        return innerIntegration_rho(rho95, T95)

    
def bound_Ts_for_Edot(logMdot,Edot_LEdd,logTsa0,logTsb0,npts_Ts=10,tol=1e-5,Verbose=0):

    # Find values of logTs that bound the true wind model that can be integrated to infinity
    # Begins with initial bound values logTsa0 and logTsb0. outerIntegration with a should 
    # reach a termination event (sol.status=1), with a should crash with stepsize->0 (sol.status=-1)

    print('\nFinding Ts for Edot/LEdd = %.10f'%Edot_LEdd)

    # If roots already exist for this Mdot, we can predict where Ts might be, and use that
    # to avoid wasting time searching where it will not be
    try:
        logMdots,roots = IO.load_roots()
        elts = [i for i in range(len(logMdots)) if logMdots[i]>logMdot] # grab Mdots larger than current one
        if len(elts)>=2:
            x1,x2 = logMdots[elts[0]], logMdots[elts[1]]
            y1,y2 = roots[elts[0]][1], roots[elts[1]][1]
            # Interpolate a line to predict logTs for our Mdot
            logTs_pred = (y2-y1)/(x2-x1) * (logMdot-x1) + y1
        else:
            logTs_pred = None 
    except:
        logTs_pred = None

    a,b = logTsa0,logTsb0
    logTsvals = np.linspace(a,b,npts_Ts)
    logTsvals = np.round(logTsvals,9)

    # Begin search
    while abs(b-a)>tol:
        print('%.8f    %.8f'%(a,b))

        for logTs in logTsvals[1:-1]:

            print('Current: %.8f'%logTs, end="\r")

            try:
                res,rs = run_outer(logMdot,Edot_LEdd,logTs,Verbose)
            except Exception as E:
                print(E)
                print('Exiting...')
                return None,None

            else:
                if res.status==1:
                    a = logTs
                elif res.status==0:
                    raise Exception('Reached end of integration interval (r=%.3e) without diverging!'%res.t[-1])
                else: # res.status=-1
                    b = logTs
                    break

        if logTs_pred is not None:
            if a - logTs_pred > 0.3:
                print('Bottom logTs (%.2f) too far from where the root will realistically be (prediction from two other Mdots is logTs=%.2f'%(a,logTs_pred))
                return None,None

        if a==b:
            print('border values equal (and did not hit rs<RNS, maybe allow higher Ts). Exiting')
            break

        logTsvals = np.linspace(a,b,6)      

    print('\nok')
    return a,b


def get_EdotTsrel(logMdot,tol=1e-5,Verbose=0,Edotmin=1.01,Edotmax=1.04,npts=10,save_decimals=8):

    # find the value of Ts that allow a solution to go to inf (to tol precision), for each value of Edot

    if Verbose: print('\nLOGMDOT = %.2f\n'%logMdot)

    Edotvals = np.linspace(Edotmin,Edotmax,npts)
    Edotvals = np.round(Edotvals,10)
    
    Tsvals = []
    cont = True

    rel = IO.load_EdotTsrel(logMdot)

    for Edot_LEdd in Edotvals:

        if rel[0] is True: #relation already exists, will use it to predict initial a,b bounds on Ts
            _,Edotrel,Tsrel,_ = rel

            ilower = [i for i in range(len(Edotrel)) if Edotrel[i]<Edot_LEdd] # gives all the indexes of all Edots which are lower than current
            iupper = [i for i in range(len(Edotrel)) if Edotrel[i]>Edot_LEdd] # higher than current

            if len(ilower)>=1: # current Edot is low-bounded
                logTsa0 = Tsrel[ilower[-1]]

                if len(iupper)>=1 : # current Edot is bounded on both sides
                    logTsb0 = Tsrel[iupper[0]]
                    npts_Ts = 20 # if we are here we are really refining the param space

                else:               # current Edot is low-bounded but not up-bounded
                    logTsb0 = logTsa0+0.5
                    npts_Ts = 10

            elif len(iupper)>=1 : # current Edot is up-bounded but not low-bounded
                logTsb0 = Tsrel[iupper[0]]
                logTsa0 = logTsb0-0.5
                npts_Ts = 10

        else:
            logTsa0,logTsb0 = 6.2,8
            npts_Ts=20

        a,b = bound_Ts_for_Edot(logMdot,Edot_LEdd,logTsa0,logTsb0,npts_Ts=npts_Ts,tol=tol,Verbose=Verbose)

        if a is None:
            break

        # Save one at a time
        IO.save_EdotTsrel(logMdot,[Edot_LEdd],[a],[b],decimals=save_decimals)
        rel = IO.load_EdotTsrel(logMdot)

    IO.clean_EdotTsrelfile(logMdot,warning=0)


def Check_EdotTsrel(logMdot, recalculate=True, near_root_only=False):

    # Suppose the EdotTsrel is slightly wrong, namely the solutions corresponding
    # to logTsa and logTsb are not binding, they both crash in the same direction.
    # This could be because we changed the EOS or NS parameters, therefore the roots
    # have moved a bit. Instead of starting from scratch, we can search around the 
    # current EdotTsrel and just adjust it. This will save a lot of time.

    IO.clean_EdotTsrelfile(logMdot, warning=0)
    _,Edotvals,TsvalsA,TsvalsB = IO.load_EdotTsrel(logMdot)

    # Option : only check for 3 values on each side of the root
    if near_root_only:

        Edotroot = IO.load_wind_roots(logMdot=logMdot)[0]        
        Edotvalsnew,TsvalsAnew,TsvalsBnew = [],[],[]

        # Edots<Edot root
        k=0
        for E in [Edot for Edot in Edotvals if Edot<Edotroot][::-1]:
            if k<4:
                Edotvalsnew.append(E)
                i = Edotvals.index(E)
                TsvalsAnew.append(TsvalsA[i])
                TsvalsBnew.append(TsvalsB[i])
            k+=1
        Edotvalsnew,TsvalsAnew,TsvalsBnew = Edotvalsnew[::-1],TsvalsAnew[::-1],TsvalsBnew[::-1]

        # Edots>Edot root
        k=0
        for E in [Edot for Edot in Edotvals if Edot>Edotroot]:
            if k<4:
                Edotvalsnew.append(E)
                i = Edotvals.index(E)
                TsvalsAnew.append(TsvalsA[i])
                TsvalsBnew.append(TsvalsB[i])
            k+=1

        Edotvals,TsvalsA,TsvalsB = Edotvalsnew,TsvalsAnew,TsvalsBnew


    print('\n Checking EdotTs rel for %d Edot values'%len(Edotvals))

    for i in range(len(Edotvals)):

        # There can be rounding errors here, so we check TsA rounded down, TsB rounded up
        decimals = len(str(TsvalsA[i])) - 2 
        TsA = TsvalsA[i] - 10**(-decimals)
        TsB = TsvalsB[i] + 10**(-decimals)
        sola,_ = run_outer(logMdot,Edotvals[i],TsA)
        solb,_ = run_outer(logMdot,Edotvals[i],TsB)

        if sola.status == 1 and solb.status == -1:
            print('\nEdotTsrel file at Edot/LEdd=%.6f is OK!'%(Edotvals[i]))
        else:
            print('\nEdotTsrel file at Edot/LEdd=%.6f needs fixing'%(Edotvals[i]))

            if recalculate is True:

                logTsa,logTsb = TsvalsA[i],TsvalsB[i]
                print('Initial bounds of logTs : %.8f   %.8f'%(logTsa,logTsb))
                if sola.status == -1:
                    while sola.status == -1:
                        logTsa -= 1e-3
                        sola,_ = run_outer(logMdot,Edotvals[i],logTsa)

                elif solb.status == 1:
                    while solb.status == 1:
                        logTsb += 1e-3
                        solb,_ = run_outer(logMdot,Edotvals[i],logTsb)

                a,b = bound_Ts_for_Edot(logMdot,Edotvals[i],logTsa,logTsb)
                print('New bounds of logTs : %.8f   %.8f'%(a,b))

                if a is not None:
                    IO.save_EdotTsrel(logMdot,[Edotvals[i]],[a],[b])

    IO.clean_EdotTsrelfile(logMdot,warning=0)


def RootFinder(logMdot,checkrel=True,Verbose=False,depth=1):

    """ Find the (Edot,Ts) pair that minimizes the error on the inner boundary condition """

    print('\nStarting root finding algorithm for logMdot = %.2f'%logMdot)

    # Check if Edot,Ts file exists
    rel = IO.load_EdotTsrel(logMdot)
    if rel[0] is False:
        print('Edot-Ts relation file does not exist, creating..')

        if logMdot >= 18.0:  # no real difficulties at high Mdot
            get_EdotTsrel(logMdot,Verbose=Verbose)
        elif logMdot < 18.0 and logMdot >= 17.2:
            # At low Mdots (~<18, high Edots go to high Ts quickly, hitting
            # the rs = RNS line and causing problems in rootfinding)
            get_EdotTsrel(logMdot,Verbose=Verbose,Edotmax=1.03)
        
        elif logMdot < 17.2:
            # low Mdots are hard, it's best to narrow the search using previous roots
            logMdots,roots = IO.load_wind_roots()
            elts = [i for i in range(len(logMdots)) if logMdots[i]>logMdot] # grab Mdots larger than current one
            x1,x2 = logMdots[elts[0]], logMdots[elts[1]]
            y1,y2 = roots[elts[0]][0], roots[elts[1]][0] # Edot values
            # Interpolate a line to predict Edot for our Mdot
            Edot_pred = (y2-y1)/(x2-x1) * (logMdot-x1) + y1
            low_bound = Edot_pred - abs(Edot_pred-y1)/4
            high_bound = Edot_pred + abs(Edot_pred-y1)/4
            # input('Predicted Edot : %.6f. Will search from %.6f to %.6f'%(Edot_pred,low_bound,high_bound))
            get_EdotTsrel(logMdot,Verbose=Verbose,Edotmin=low_bound,Edotmax=high_bound, npts=5)



        rel = IO.load_EdotTsrel(logMdot)
        print('\nDone!')

    
    if Verbose: print('Loaded Edot-Ts relation from file')
    _,Edotvals,TsvalsA,TsvalsB = rel

    # Check if file is correct, i.e the two Ts values diverge in different directions
    if checkrel:
        print('Checking if relation is correct')
        for i in (0,-1):
            sola,_ = run_outer(logMdot,Edotvals[i],TsvalsA[i])
            solb,_ = run_outer(logMdot,Edotvals[i],TsvalsB[i])
            if sola.status == solb.status:
                print('Problem with EdotTsrel file at Edot/LEdd=%.8f ,logTs=%.8f'%(Edotvals[i],TsvalsA[i]))
                print(sola.message)
                print(solb.message)

        print(' EdotTsrel file ok')

    # Now do a 1D search on the interpolated line based on the inner BC error
    if len(Edotvals)<=3: # need at least 4 points for cubic spline
        print('Not enough points to interpolate spline, re-interpolating')
        if len(Edotvals)==1:
            get_EdotTsrel(logMdot,Edotmin=Edotvals[0]-0.001,Edotmax=Edotvals[0]+0.001,npts=5)
        else:
            diff = Edotvals[-1]-Edotvals[0]
            get_EdotTsrel(logMdot,Edotmin=Edotvals[0]+0.2*diff,Edotmax=Edotvals[-1]-0.2*diff,npts=3)
        raise Exception('Call Again')

    else:
        rel_spline = IUS(Edotvals,TsvalsA)
    
    def Err(Edot_LEdd):
        if isinstance(Edot_LEdd,np.ndarray): Edot_LEdd=Edot_LEdd[0]
        logTs=rel_spline(Edot_LEdd)
        E = run_inner(logMdot,Edot_LEdd,logTs)
        print("Looking for root... Edot/LEdd=%.6f \t logTs=%.6f \t Error= %.6f"%(Edot_LEdd,logTs,E),end="\r")
        # print("Looking for root... Edot/LEdd=%.10f \t logTs=%.10f \t Error= %.10f"%(Edot_LEdd,logTs,E))
        return E


    if Verbose: print('Searching root on Edot,Ts relation based on inner boundary error')

    flag_300 = (False,)
    erra = Err(Edotvals[0])

    for Edot in Edotvals[:0:-1]: # cycle through the values in reverse order
        errb = Err(Edot)

        if errb == -300: # a special error we might have to deal with
            flag_300 = (True, Edotvals.index(Edot))  

        elif erra*errb < 0: # different sign, means a root is in the interval
            print('\nroot present')
            break

    if erra*errb > 0: # same sign (we'll enter this if we didn't break last loop)

        if erra<0:
            print('\nOnly negative errors (rb<RNS)') # need smaller Ts (smaller Edot)
            diff = Edotvals[1]-Edotvals[0]
            get_EdotTsrel(logMdot,Edotmin=Edotvals[0]-1e-3,Edotmax=Edotvals[0]-1e-8,npts=2*depth)

        else:

            if not flag_300[0]:
                print('\nOnly positive errors (rb>RNS)') # need higher Ts (higher Edot)
                diff = Edotvals[-1]-Edotvals[-2]
                get_EdotTsrel(logMdot,Edotmin=Edotvals[-1]+1e-8,Edotmax=Edotvals[-1]+diff,npts=2*depth)

            else:
                print('\nThe only negative errors (rb<RNS) don''t converge, need to refine')
                i = flag_300[1]
                diff = Edotvals[i]-Edotvals[i-1]
                get_EdotTsrel(logMdot,Edotmin=Edotvals[i-1]+0.25*diff,Edotmax=Edotvals[i-1]+0.75*diff,npts=2*depth,tol=1e-9,save_decimals=10)


        raise Exception('Call Again')

    else:
        x = brentq(Err,Edotvals[0],Edot) # Edot is the last value before break above
        root = [x,rel_spline(x).item(0)]
        print('Found root : ',root,'. Error on NS radius: ',Err(x))
        return root


def ImproveRoot(logMdot, eps=0.1, npts=5):

    ''' Rootfinding with the inner b.cond is done on a spline interpolation of 
    the Edot-Ts relation. When that root is obtained, it's not exact because of 
    interpolation errors, meaning that root can't be carried out to infinity. 
    In the main code (wind_GR_FLD), a new Ts bound is found to do the bisection,
    which changes the sonic point. At low mdot (or in general when the Edot-Ts)
    relation doesn't have enough points, the change in sonic point can be 
    significant enough that the base (r(y8)) is not close at all to RNS anymore.
    The purpose of this function is to resolve the Edot-Ts relation around the 
    root found initially and then re-do the rootfinding.
    '''

    print('\n\n ** Improving root for logMdot = %.2f ** '%logMdot)

    logMdots,roots = IO.load_wind_roots()
    if logMdot not in logMdots:
        sys.exit("root doesn't exist")
    root = roots[logMdots.index(logMdot)]

    if logMdot<17.25:
        decimals,tol = 10,1e-10
    else:
        decimals,tol = 8,1e-8

    # Search between the two Edot values that bound the root
    _,Edots,_,_ = IO.load_EdotTsrel(logMdot)
    i = np.argwhere(np.array(Edots)>root[0])[0][0]
    diff = Edots[i]-Edots[i-1]
    bound1 = Edots[i-1] + eps*diff
    bound2 = Edots[i] - eps*diff

    get_EdotTsrel(logMdot,Edotmin=bound1,Edotmax=bound2,npts=npts,tol=tol,save_decimals=decimals)
    #get_EdotTsrel(logMdot,Edotmin=root[0]-eps,Edotmax=root[0]+eps,npts=8)
    IO.clean_EdotTsrelfile(logMdot,warning=0)
    root = RootFinder(logMdot,checkrel=False)
    IO.save_wind_root(logMdot,root,decimals=decimals)
    IO.clean_wind_rootfile(warning=0)



###################################### Driver ########################################

def recursor(logMdot, depth=1, max_depth=5):
    # will call RootFinder many times by recursion, as long as it returns the "Call Again" exception

    if depth==max_depth:
        print('Reached max recursion depth')
        return None 

    try:
        root = RootFinder(logMdot,checkrel=False,depth=depth)
        return root

    except Exception as E:

        if E.__str__() == 'Call Again':
            print('\nGoing into recursion depth %d'%(depth+1))
            root = recursor(logMdot, depth=depth+1)
            return root

        else:
            print(E)
            raise Exception('Problem')


def driver(logmdots):

    if logmdots == 'all':
        logmdots = IO.load_wind_roots()[0]

    elif logmdots == 'all_models':
        logmdots = IO.get_wind_list()[::-1]

    elif type(logmdots) == float or type(logmdots) == int:
        logmdots = [logmdots]

    success, max_recursed, problems = [],[],[]

    for logMdot in logmdots:

        try: 
            root = recursor(logMdot)

            if root is None:
                max_recursed.append(logMdot)

            else:
                success.append(logMdot)
                if logMdot<17:
                    IO.save_wind_root(logMdot,root,decimals=10)
                else:
                    IO.save_wind_root(logMdot,root)

        except Exception as e:
                problems.append(logMdot)
                print('\n',e)
                print('\nPROBLEM WITH LOGMDOT = ',logMdot,'\nTrying again with verbose and checking EdotTs rel...\n\n')
                try : RootFinder(logMDOT,checkrel=True,Verbose=True)
                except: pass


    print('\n\n*********************  SUMMARY *********************')
    print('Found roots for :',success)
    print('Reached recursion limit trying to find roots for :',max_recursed)
    print('There were problems for :',problems)

    if len(success)>=1: #and input('\nClean (overwrite) updated root file? (0 or 1) '):
       IO.clean_wind_rootfile(warning=0)
    


# Command line call
if __name__ == "__main__":
    if len(sys.argv)>1:
        
        if sys.argv[1]!='all' and ' ' in sys.argv[1]:          
            sys.exit('Give logmdots as a,b,c,...')

        if sys.argv[1]=='all':
            logmdots='all'
        elif sys.argv[1]=='all_models':
            logmdots='all_models'
        elif ',' in sys.argv[1]:
            logmdots = eval(sys.argv[1])
        else:
            logmdots = [eval(sys.argv[1])]

        driver(logmdots)

            
            

