import sys
import IO

from envelopes.env_GR_FLD import MakeEnvelope
from winds.RootFinding import driver
from winds.wind_GR_FLD import MakeWind


# Toggle verbose output
Verbose = True


def run_envelopes(Rphotkm_list):

    problems,success = [],[]

    for rph in Rphotkm_list:
        print('\n*** Calculating envelope with photosphere at %f km***\n'%rph)
        try:
            env = MakeEnvelope(rph,Verbose=Verbose)
            success.append(rph)
            IO.write_envelope(rph,env)
            print('Rphot=%s done'%str(rph))

        except Exception as E:
            if E.__str__() == 'Call Again':
                run_envelopes((rph,))
            else:
                problems.append(rph)
                print('PROBLEM WITH Rphot = %.3f km'%rph)
        
    print('\n\n*********************  SUMMARY *********************')
    print('Found solutions for these values : ',success)
    if len(problems)>=1:
        print('There were problems for these values : ',problems,'\n')



def run_wind(logMdot_list=IO.load_wind_roots()[0][::-1]):

    # Roots
    logMdot_get_root = [logMdot for logMdot in logMdot_list if logMdot not in IO.load_wind_roots()[0]]
    if len(logMdot_get_root)>0:
        print('*** Finding roots for winds ***')
        driver(logMdot_get_root)


    # Winds for logMdots with succesful roots
    logMdots_roots,_ = IO.load_wind_roots()
    problems,success = [],[]

    for logMdot in logMdot_list:

        if logMdot in logMdots_roots:
            root = IO.load_wind_roots(logMdot)

            print('\n*** Calculating wind with logMdot = %.2f km***\n'%logMdot)
            try:
                wind = MakeWind(root,logMdot,Verbose=Verbose)
                success.append(logMdot)
                IO.write_wind(logMdot, wind)
                print('logMdot=%s done'%str(logMdot))

            except Exception as E:
                print(E)
                problems.append(logMdot)
                print('PROBLEM WITH logMdot = %.2f km'%logMdot)
        
    print('\n\n*********************  SUMMARY *********************')
    print('Found solutions for these values : ',success)
    if len(problems)>=1:
        print('There were problems for these values : ',problems,'\n')






## Command-line call 

err_messages = [
    'First argument must be envelope or wind',
    'Give photospheric radii in km, or mass-loss rates as log10 values, separated by commas or spaces',
    'RNS<rph<100',
    '16<logMdot<19']

if __name__ == '__main__':

    IO.make_directories()

    if len(sys.argv) >= 2:

        try:
            if len(sys.argv)>3:
                l = [eval(s) for s in sys.argv[2:]]
            else:
                l = [eval(s) for s in sys.argv[2].split(',')]
        except:
            if sys.argv[1] not in ('envelope','wind'):
                sys.exit(err_messages[0])
            else:
                sys.exit(err_messages[1])

        if sys.argv[1] == 'envelope':
            if min(l)<IO.load_params()['R'] or max(l)>100:
                sys.exit(err_messages[2])
            else:
                run_envelopes(l)

        elif sys.argv[1] == 'wind':
            if min(l)<16 or max(l)>19:
                sys.exit(err_messages[3])
            else:
                run_wind(l)

        else:
            sys.exit(err_messages[0])