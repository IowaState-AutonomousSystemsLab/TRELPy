from pathlib import Path
from experiment_file import *
from controllers import construct_controllers as K_des
from formula import *
from problem_setup import *
import json

def synthesize(MAX_V):
    Ncar = get_Ncar(MAX_V=MAX_V)
    for vmax in range(1,MAX_V+1):
        print("===========================================================")
        print("Max Velocity: ", vmax)
        # Initial conditions set for all velocities
        Vlow, Vhigh, xped, formula = initialize(vmax, Ncar)
        print("Specification: ")
        print(formula)
        K = dict() # Dictionary of all controllers across speeds
        for vcar in range(1, vmax+1):  # Initial speed at starting point
            K[vcar] = dict()
            control_dir_for_speed = control_dir + ""
            K[vcar] = K_des.construct_controllers(Ncar, Vlow, Vhigh, xped, vcar,control_dir=control_dir_for_speed)
        
        # control_dict_file = control_dir + f"/controllers_for_max_speed_{Vhigh}.json"
        # with open(control_dict_file, 'w') as f:
        #     f.dump(K)

if __name__=="__main__":
    MAX_V = 3
    synthesize(MAX_V)