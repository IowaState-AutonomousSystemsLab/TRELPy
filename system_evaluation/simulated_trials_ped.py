### Simulated Car Trials
# Driving down a straight line
# Apurva Badithela

import numpy as np
import random
from pdb import set_trace as st
from controllers.simple_controller import control_dict, prop_control_dict
import control as ct
from experiment_file import *
import  json
from formula import *
from print_utils import print_cm, print_param_cm

try: 
    from system_evaluation.simple_markov_chain import construct_mc as cmp
    from system_evaluation.simple_markov_chain.setup_mc import call_MC, call_MC_param
except:
    from simple_markov_chain import construct_mc as cmp
    from simple_markov_chain.setup_mc import call_MC, call_MC_param

#################################################
# Car Dynamics. Car state: [x, y, theta]
# KinCar: Simple Car dynamics. Car driving in straight line along Y-axis and zero-order hold controller. 
# DiffFlatCar: Implementing differentially flat controllers from the Python control toolbox.
class KinCar:
    def __init__(self, x_init, u_init, L=3):
        self.x = x_init
        self.u_init = u_init
        self.ydot = self.u_init[0]*np.sin(self.x[2])
        self.L = L # Wheel base
        pass

    def vehicle_update(self, us, uphi):
        dx = np.array([np.cos(self.x[2]) * us, np.sin(self.x[2]) * us, (us/self.L) * np.tan(uphi)])
        return dx

    def simulate_step(self, us, uphi, dt=0.1):
        dx = self.vehicle_update(us, uphi)
        self.x += dx*dt       
        self.ydot = us*np.sin(self.x[2])

class DiffFlatCar:
    def __init__(self, x_init, L=3):
        self.x = x_init
        self.L = L # Wheel base
        pass

    def vehicle_update(self, us, uphi):
        dx = np.array([np.cos(self.x[2]) * us, np.sin(self.x[2]) * us, (us/self.L) * np.tan(uphi)])
        return dx

    def simulate_step(self, us, uphi, dt):
        dx = self.vehicle_update(us, uphi)
        self.x += dx*dt        

#################################################
# Parameters
d2c = 10 # Factor by which discrete measurement converts to continuous units
c2d = 1.0/d2c # Needs to be further approximated to get into integer form.
dt = 0.1 # seconds. Each individual update step
x0 = np.array([0, 0, np.pi/2]) # Driving up
u0 = np.array([1, 0])
image_rate = 0.1 # Hz.
Ntrials = 10000

# Helper functions to convert discrete to continuous and vice-versa
def dis_to_cont(x_abs, v_abs):
    x = (x_abs - 1)*d2c
    v = v_abs # Usually not important
    return x, v

def cont_to_dis(x, v):
    x_abs = int(np.ceil(x*c2d))
    v_abs = np.ceil(v)
    return x_abs, v_abs

# Radius band from a discrete value:
def get_radius_band(x_abs, x_ped):
    z = (abs(x_ped-x_abs)//10) # Double check this logic. DOesn't make sense for larger distances
    ld = z*d2c
    ud = ld + d2c
    ld += 1
    band = (ld, ud)
    return band
#################################################
# Get COnfusion Matrix
def get_confusion_matrix():
    C, param_C = cmp.confusion_matrix(cm_fn)
    return C, param_C

#################################################
# Sample observation according to confusion matrix
def sample(C, true_env_type):
    C_true_env = {k: v for k,v in C.items() if k[1]==true_env_type}
    cdf = 0
    rand_obs = np.random.rand()
    for k, v in C_true_env.items():
        cdf += v
        if rand_obs <= cdf:
            obs = k[0]
            break
    return obs

# Sample observation from distance parametrized confusion matrix
def sample_param(C, x_abs, xped, true_env_type):
    radius_band = get_radius_band(x_abs, xped)
    C_true_env = {k: v for k,v in C[radius_band].items() if k[1]==true_env_type}
    cdf = 0
    rand_obs = np.random.rand()
    for k, v in C_true_env.items():
        cdf += v
        if rand_obs <= cdf:
            obs = k[0]
            break
    return obs

#################################################
# Simulate trials
# Each trial is a random run of the car starting from origin.
# Result: 0 means failure and 1 means success.
def trial(x_init_abs, v_init_abs, K_strat, xped, C, O, class_dict, true_env_type, param=False):
    x0, u0 = dis_to_cont(x_init_abs, v_init_abs)
    xped_cont, _ = dis_to_cont(xped, 0)
    x_cw_cont, _ = dis_to_cont(xped-1, 0)
    car = KinCar(np.array([0, x0, np.pi/2]), np.array([u0, 0]))
    traj = [(car.x[1], car.ydot)]
    x_curr_abs = x_init_abs
    v_curr_abs = v_init_abs
    
    if param:
        obs = sample_param(C, x_curr_abs, xped, true_env_type=true_env_type)
    else:
        obs = sample(C, true_env_type=true_env_type)
    # obs = true_env_type
    assert obs in O

    while car.x[1] < xped_cont:
        # Control law
        (trg_x_abs, trg_v_abs) = K_strat[obs][(x_curr_abs, v_curr_abs)]  # Discrete
        uphi = 0                                               # Continuous
        us = trg_v_abs

        # Simulate step
        car.simulate_step(us, uphi, dt=0.1)

        # Observe
        x_abs, v_abs = cont_to_dis(car.x[1], car.ydot)

        if x_abs > x_curr_abs:
            if param:
                obs = sample_param(C, x_curr_abs, xped, true_env_type=true_env_type)
            else:
                obs = sample(C, true_env_type=true_env_type)
            # obs = true_env_type
            x_curr_abs = x_abs
            v_curr_abs = v_abs

        # Break if violating requirement or on test completion. 
        # Trial is the only change to get the results.
        if car.ydot == 0 and car.x[1] < x_cw_cont:
            result = 0
            break
        elif car.ydot > 0 and car.x[1] >= x_cw_cont:
            result = 0
            break
        elif car.ydot == 0 and car.x[1] >= x_cw_cont:
            result = 1
            break
        else:
            result = 1
    return result

#################################################
# Simulate trials
# Each trial is a random run of the car starting from origin.
def init(MAX_V=6):
    Ncar = int(MAX_V*(MAX_V+1)/2 + 4)
    return Ncar

def save_results(INIT_V, P, P_param, result_type, true_env):
    results_folder = f"{cm_dir}/simulated_probability_results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    fname_v = Path(f"{results_folder}/{result_type}_cm_{true_env}_vmax_"+str(MAX_V)+"_initv.json")
    fname_p = Path(f"{results_folder}/{result_type}_cm_{true_env}_vmax_"+str(MAX_V)+"_prob.json")
    fname_param_p = Path(f"{results_folder}/{result_type}_param_cm_{true_env}_vmax_"+str(MAX_V)+"_prob.json")

    #pdb.set_trace()
    with open(fname_v, 'w') as f:
        json.dump(INIT_V, f)
    with open(fname_p, 'w') as f:
        json.dump(P, f)
    with open(fname_param_p, 'w') as f:
        json.dump(P_param, f)

def initialize(MAX_V, Ncar, maxv_init=None):
    '''
    Inputs::
    MAX_V: Maximum speed that the car can travel at
    Ncar: Maximum discrete states for the car
    vmax_init: Max initial speed of the car (specified if different from MAX_V)

    Outputs::
    Vlow: Minimum car speed (0)
    Vhigh: Maximum car speed (MAX_V)
    xped: Pedestrian position
    '''

    Vlow = 0
    Vhigh = MAX_V
    
    if maxv_init:
        xmax_stop = maxv_init*(maxv_init+1)/2 + 1 # earliest stopping point for car 
    else:
        xmax_stop = Vhigh*(Vhigh+1)/2 + 1 # earliest stopping point for car 
    
    xped, xcar_stop = set_crosswalk_cell(Ncar, xmax_stop)
    return Vlow, Vhigh, xped

def simulate(MAX_V=6):
    Ncar = init(MAX_V=MAX_V)
    C, param_C = cmp.confusion_matrix(cm_fn)
    print(" =============== Full confusion matrix ===============")
    print_cm(C)
    print(" =============== Parametrized confusion matrix ===============")
    print_param_cm(param_C)
    print("===========================================================")
    INIT_V, P, P_param = simulated_probabilities(Ncar, MAX_V, C, param_C,true_env_type="ped")
    save_results(INIT_V, P, P_param, "class", "ped")

def simulated_probabilities(Ncar, MAX_V, C, param_C, true_env_type="ped"):
    INIT_V = []
    P = []
    P_param = []
    
    Vlow, Vhigh, xped = initialize(MAX_V, Ncar)

    O = {"ped", "obs", "empty"}
    class_dict = {0: {'ped'}, 1: {'obs'}, 2: {'empty'}}

    for vcar in range(1, MAX_V+1):  # Initial speed at starting point
        x_init_abs = 1
        v_init_abs = vcar
        
        result_sum = 0
        result_param_sum = 0

        K_strat = control_dict(Ncar, Vhigh, O, xped)
        for k in range(Ntrials):    # Trials
            result = trial(x_init_abs, v_init_abs, K_strat, xped, C, O, class_dict, true_env_type)
            result_param = trial(x_init_abs, v_init_abs, K_strat, xped, param_C, O, class_dict, true_env_type, param=True)

            result_sum += result
            result_param_sum += result_param

        success = result_sum/Ntrials
        success_param = result_param_sum/Ntrials

        P.append(success)
        P_param.append(success_param)

        print('Probability of eventually reaching good state for initial speed, {}, and max speed, {} is p = {}:'.format(vcar, MAX_V, success))
        # Store results:
        INIT_V.append(vcar)
    return INIT_V, P, P_param

if __name__=="__main__":
    MAX_V = 6
    simulate(MAX_V=MAX_V)