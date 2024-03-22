from pathlib import Path
from experiment_file import *
from controllers import construct_controllers as K_des
from formula import *
from problem_setup import *

def synthesize(MAX_V):
    Ncar = get_Ncar(MAX_V=MAX_V)
    for vmax in range(1,MAX_V+1):
        print("===========================================================")
        print("Max Velocity: ", vmax)
        # Initial conditions set for all velocities
        Vlow, Vhigh, xped, formula = initialize(vmax, Ncar)
        print("Specification: ")
        print(formula)
        for vcar in range(1, vmax+1):  # Initial speed at starting point
            state_f = lambda x,v: (Vhigh-Vlow+1)*(x-1) + v
            start_state = "S"+str(state_f(1,vcar))
            print(start_state)

            S, state_to_S, K_backup = cmp.system_states_example_ped(Ncar, Vlow, Vhigh)
            control_dir_for_speed = control_dir + ""
            K = K_des.construct_controllers(Ncar, Vlow, Vhigh, xped, vcar,control_dir=control_dir_for_speed)

if __name__=="__main__":
    MAX_V = 3
    synthesize(MAX_V)