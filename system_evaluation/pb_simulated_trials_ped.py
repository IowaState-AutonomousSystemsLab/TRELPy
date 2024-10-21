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
import networkx as nx

try: 
    from system_evaluation.simple_markov_chain import prop_construct_mc as cmp
    from system_evaluation.simple_markov_chain.setup_prop_mc import call_MC, call_MC_param
except:
    from simple_markov_chain import prop_construct_mc as cmp
    from simple_markov_chain.setup_prop_mc import call_MC, call_MC_param

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

    def simulate_step(self, us, uphi, dt = 0.1):
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
d2c = 10 # Factor by which discrete measurement converts to continuous units; only for radius band interpretation
c2d = 1.0/d2c # Needs to be further approximated to get into integer form.
dt = 0.1 # seconds. Each individual update step
x0 = np.array([0, 0, np.pi/2]) # Driving up
u0 = np.array([1, 0])
Ntrials = 100000

# Helper functions to convert discrete to continuous and vice-versa
def dis_to_cont(x_abs, v_abs):
    #x = (x_abs-1)*d2c
    x = (x_abs-1)
    v = v_abs # Usually not important
    return x, v

def cont_to_dis(x, v):
    #x_abs = int(np.ceil(x*c2d))
    x_abs = int(np.ceil(x))
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
# Get Confusion Matrix
def get_confusion_matrix():
    C, param_C = cmp.confusion_matrix(cm_fn)
    return C, param_C

#################################################
# Sample observation according to confusion matrix
def sample(C, prop_dict, true_env_type):
    C_true_env = dict()
    for idx, obs_tuple in prop_dict.items():
        if {true_env_type} == obs_tuple:
            C_true_env = {k: v for k,v in C.items() if k[1]==idx}
    
    cdf = 0
    rand_obs = np.random.rand()
    for k, v in C_true_env.items():
        cdf += v
        if rand_obs <= cdf:
            obs = prop_dict[k[0]]
            break
    return obs

def debug_sample(C, prop_dict, true_env_type):
    C_true_env = dict()
    for idx, obs_tuple in prop_dict.items():
        if {true_env_type} == obs_tuple:
            C_true_env = {k: v for k,v in C.items() if k[1]==idx}
    
    C_sampled_true_env = dict()
    C_sampled_true_env = {k: 0 for k in C_true_env.keys()}
    for n in range(Ntrials):
        cdf = 0
        rand_obs = np.random.rand()
        for k, v in C_true_env.items():
            cdf += v
            if rand_obs <= cdf:
                C_sampled_true_env[k] += 1
                break
    
    C_sampled_true_env = {k: v/Ntrials for k,v in C_sampled_true_env.items()}
    return C_sampled_true_env

# Sample observation from distance parametrized confusion matrix
def sample_param(C, x_abs, xped, prop_dict, true_env_type):
    radius_band = get_radius_band(x_abs, xped)
    C_true_env = dict()
    for idx, obs_tuple in prop_dict.items():
        if {true_env_type} == obs_tuple:
            C_true_env = {k: v for k,v in C[radius_band].items() if k[1]==idx}
    cdf = 0
    rand_obs = np.random.rand()
    for k, v in C_true_env.items():
        cdf += v
        if rand_obs <= cdf:
            obs = prop_dict[k[0]]
            break        
    return obs

#################################################
def cont_control(us_new, uphi_new, x, v, trg_x_abs, trg_v_abs):
    # dt = 0.1
    # Parameter chosen for problem. Must be any value >= (yd-y0)/dt -v0-vd + 1
    if np.ceil(v) == trg_v_abs:
        return us_new, uphi_new
    else:
        trg_x_cont, trg_v_cont = dis_to_cont(trg_x_abs, trg_v_abs)
        acc = (trg_v_cont**2 - v**2)/(2*(trg_x_cont - x))
        us = v + acc*dt
        us = max(min(us, np.ceil(v)), trg_v_abs)
        return us, uphi_new
    
#################################################
# Simulate trials
# Each trial is a random run of the car starting from origin.
# Result: 0 means failure and 1 means success.
def trial(x_init_abs, v_init_abs, K_strat, xped, C, O, prop_dict, true_env_type, G= None, mc=None, state_to_S=None, param=False):
    x0, u0 = dis_to_cont(x_init_abs, v_init_abs)
    xped_cont, _ = dis_to_cont(xped, 0)
    x_cw_cont, _ = dis_to_cont(xped-1, 0)
    car = KinCar(np.array([0, x0, np.pi/2]), np.array([u0, 0]))
    traj = [(car.x[1], car.ydot)]
    x_curr_abs = x_init_abs
    v_curr_abs = v_init_abs
    debug_sample(C, prop_dict, true_env_type)
    if param:
        obs = sample_param(C, x_curr_abs, xped, prop_dict, true_env_type=true_env_type)
    else:
        obs = sample(C, prop_dict, true_env_type=true_env_type)
    
    for control_obs, _ in K_strat.items():
        if isinstance(control_obs, tuple):
            if set({control_obs}) == obs:
                break
        else:
            if set({control_obs,}) == obs:
                break

    (trg_x_abs, trg_v_abs) = K_strat[control_obs][(x_curr_abs, v_curr_abs)]  # Discrete
    
    assert control_obs in O

    count=0
    edge = ((x_curr_abs,v_curr_abs), (trg_x_abs, trg_v_abs))
    
    while car.x[1] < xped_cont:
        # Discrete
        uphi_new = 0                                               # Continuous
        us_new = trg_v_abs
        us, uphi = cont_control(us_new, uphi_new, car.x[1], car.ydot, trg_x_abs, trg_v_abs)
            
        epsilon = 0.05
        u_eps = 0.1

        car.simulate_step(us, uphi, dt = 0.1)
        
        # If car has transitioned to the next cell, set its target speed:
        trg_x_cont, trg_v_cont = dis_to_cont(trg_x_abs, trg_v_abs)

        if trg_x_cont - car.x[1]>=0 and trg_x_cont - car.x[1] < epsilon:
            car.ydot = trg_v_cont
            car.x[1] = trg_x_cont
        
        x_abs, v_abs = cont_to_dis(car.x[1], car.ydot)

        if x_abs == trg_x_abs:
            if edge not in G.edges():
                G.add_edge(edge[0], edge[1], weight=1.0)
            else:
                G[edge[0]][edge[1]]['weight'] += 1.0

            if v_abs != trg_v_abs:
                st()
            
            if param:
                obs = sample_param(C, x_curr_abs, xped, prop_dict, true_env_type=true_env_type)
            else:
                obs = sample(C, prop_dict, true_env_type=true_env_type)
            # obs = true_env_type
            x_curr_abs = x_abs
            v_curr_abs = v_abs
            
            for control_obs, _ in K_strat.items():
                if isinstance(control_obs, tuple):
                    if set({control_obs}) == obs:
                        break
                else:
                    if set({control_obs,}) == obs:
                        break
            (trg_x_abs, trg_v_abs) = K_strat[control_obs][(x_curr_abs, v_curr_abs)]
            edge = ((x_curr_abs,v_curr_abs), (trg_x_abs, trg_v_abs))
        # Break if violating requirement or on test completion. 
        # Trial is the only change to get the results.
        if car.ydot == 0 and car.x[1] < x_cw_cont:
            result = 0
            break
        elif car.ydot > 0 and car.x[1] >= xped_cont:
            result = 0
            break

        elif car.ydot == 0 and car.x[1] >= x_cw_cont:
            result = 1
            break
        else:
            result = 1
        count += 1
        if count >= 10000:
            st()

    return result, G

#################################################
# Simulate trials
# Each trial is a random run of the car starting from origin.
def init(MAX_V=6):
    Ncar = int(MAX_V*(MAX_V+1)/2 + 4)
    return Ncar

def save_results(INIT_V, P, P_param, result_type, true_env):
    results_folder = f"{cm_dir}/simulated_probability_results_v1"
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

def simulate_prop(MAX_V=6):
    Ncar = init(MAX_V=MAX_V)
    C, param_C, prop_dict = cmp.confusion_matrix(prop_cm_fn, prop_dict_file)
    print(" =============== Proposition-based Full confusion matrix ===============")
    print_cm(C)
    print(" =============== Parametrized Proposition-based confusion matrix ===============")
    print_param_cm(param_C)
    print("===========================================================")
    INIT_V, P, P_param = simulated_probabilities(Ncar, MAX_V,C, param_C, prop_dict,true_env_type="ped")
    save_results(INIT_V, P, P_param, "prop", "ped")


def simulate_prop_seg(MAX_V=6):
    Ncar = init(MAX_V=MAX_V)
    C, param_C, prop_dict = cmp.confusion_matrix(prop_cm_seg_fn, prop_dict_file)
    print(" =============== Segmented Proposition-based Full confusion matrix ===============")
    print_cm(C)
    print(" =============== Segmented Parametrized Proposition-based confusion matrix ===============")
    print_param_cm(param_C)
    print("===========================================================")
    
    INIT_V, P, P_param = simulated_probabilities(Ncar, MAX_V,C, param_C, prop_dict,true_env_type="ped")
    save_results(INIT_V, P, P_param, "prop_seg", "ped")

def simulated_probabilities(Ncar, MAX_V, C, param_C, prop_dict, true_env_type="ped"):
    INIT_V = []
    P = []
    P_param = []
    
    Vlow, Vhigh, xped = initialize(MAX_V, Ncar)
    print("===========================================================")

    O = ["ped", "obs", ("ped","obs"), "empty"]
    class_dict = {0: {'ped'}, 1: {'obs'}, 2: {'empty'}}
    
    for vcar in range(1, MAX_V+1):  # Initial speed at starting point
        x_init_abs = 1
        v_init_abs = vcar
        
        result_sum = 0
        result_param_sum = 0

        G = nx.DiGraph()

        #### Debug
        state_f = lambda x,v: (Vhigh-Vlow+1)*(x-1) + v
        start_state = "S"+str(state_f(1,vcar))
        state_info = dict()
        state_info["start"] = start_state
        S, state_to_S = cmp.system_states_example_ped(Ncar, Vlow, Vhigh)
        true_env = str(1)
        # M = call_MC(S, O, state_to_S, C, class_dict, true_env, true_env_type, state_info, Ncar, xped, Vhigh)
        # param_M = call_MC_param(S, O, state_to_S, param_C, class_dict, true_env, true_env_type, state_info, Ncar, xped, Vhigh)
        G.add_nodes_from(list(state_to_S.keys()))
    
        K_strat = prop_control_dict(Ncar, Vhigh, O, xped)
        for k in range(Ntrials):    # Trials
            result, G = trial(x_init_abs, v_init_abs, K_strat, xped, C, O, prop_dict, true_env_type, G=G)
            # result_param = trial(x_init_abs, v_init_abs, K_strat, xped, param_C, O, prop_dict, true_env_type, param=True)

            # result = trial(x_init_abs, v_init_abs, K_strat, xped, C, O, prop_dict, true_env_type, M, state_to_S)
            # result_param = trial(x_init_abs, v_init_abs, K_strat, xped, param_C, O, prop_dict, true_env_type, param_M, state_to_S, param=True)

            result_sum += result
            # result_param_sum += result_param

        success = result_sum/Ntrials
        #success_param = result_param_sum/Ntrials

        P.append(success)
        # P_param.append(success_param)

        print('Probability of eventually reaching good state for initial speed, {}, and max speed, {} is p = {}:'.format(vcar, MAX_V, success))
        # Store results:
        INIT_V.append(vcar)
        st()
    return INIT_V, P, P_param

if __name__=="__main__":
    MAX_V = 3
    simulate_prop(MAX_V=MAX_V)
    simulate_prop_seg(MAX_V=MAX_V)