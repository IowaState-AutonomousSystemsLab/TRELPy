# Alternative version of simple controller in which approximation takes place after.
from pdb import set_trace as st
import numpy as np

DIS_TO_CONT = 10 # Factor by which discrete measurement converts to continuous
CONT_TO_DIS = 1/DIS_TO_CONT # Needs to be further approximated to get into integer form.

def get_discrete_cell(self, xcar):
    # 1/30/24: TODO: Fix this logic. The distance bins need to be changed / fixed.
    ped_cell = ped_st[0]
    init_cell = init_st[0]
    distance_z = (abs(ped_cell-init_cell)//10)
    ld = int(distance_z*10)
    ud = int(ld + 10)
    ld += 1
    distbin = (ld, ud)
    return distbin

def convert_to_continuous(state):
    xcar, vcar = state

# Equations of motion
def control(xped, state, obs, Ncar):
    '''
    xped: Location of pedestrian
    xcar: Location of car
    vcar: Speed of car
    obs: (ped/obs/empty) type of obstacle
    '''
    xcar, vcar = state
    xcar_stop = xped - 1 # Crosswalk location
    if obs == "ped":
        steps_to_stop = int(np.ceil((vcar*vcar)/2))
        steps_to_crosswalk = xcar_stop - xcar
        
        if vcar >= 1 and steps_to_crosswalk >= 0:
            if steps_to_stop + 1 < steps_to_crosswalk:
                # Continue at same speed
                xcar_new = min(xcar + vcar, Ncar)
                vcar_new = vcar
            elif steps_to_stop + 1 >= steps_to_crosswalk:
                # Decelerate to stop
                xcar_new = min(xcar + vcar -0.5, Ncar)
                vcar_new = vcar - 1
        else:   # Not moving
            xcar_new = xcar
            vcar_new = vcar
            
        new_state = (int(np.ceil(xcar_new)), int(vcar_new))

    if obs != "ped":
        xcar_new = min(xcar + vcar, Ncar)
        new_state = (xcar_new, vcar)
    return new_state

def control_simple(xped, state, obs, Ncar):
    '''
    xped: Location of pedestrian
    xcar: Location of car
    vcar: Speed of car
    obs: (ped/obs/empty) type of obstacle
    '''
    xcar, vcar = state
    xcar_stop = xped - 1 # Crosswalk location
    
    if obs == "ped":
        steps_to_stop = int(np.ceil((vcar*vcar)/2))
        threshold_cell = xcar_stop - steps_to_stop
        steps_to_crosswalk = xcar_stop - xcar
        
        if vcar >= 1 and steps_to_crosswalk >= 0:
            if xcar + vcar -0.5 <= threshold_cell:
                # Continue at same speed
                xcar_new = min(xcar + vcar, Ncar)
                vcar_new = vcar
            elif xcar + vcar -0.5 > threshold_cell:
                # Decelerate to stop
                xcar_new = min(xcar + vcar - 0.5, Ncar)
                vcar_new = vcar - 1
        else:   # Not moving
            xcar_new = xcar
            vcar_new = vcar
            
        new_state = (int(np.ceil(xcar_new)), int(vcar_new))

    if obs != "ped":
        xcar_new = min(xcar + vcar, Ncar)
        new_state = (xcar_new, vcar)
    return new_state

def control_dict(Ncar, Vhigh, env_obs, xped):
    K = dict()
    for obs in env_obs:
        K[obs] = dict()
        for xcar in range(1, Ncar+1):
            for vcar in range(0, Vhigh+1):
                state = (xcar, vcar)
                K[obs][state] = control_simple(xped, state, obs, Ncar)
    return K

def prop_control_dict(Ncar, Vhigh, env_obs, xped):
    K = dict()
    for obs in env_obs:
        K[obs] = dict()
        for xcar in range(1, Ncar+1):
            for vcar in range(0, Vhigh+1):
                state = (xcar, vcar)
                if "ped" in obs:
                    control_for_obs="ped"                    
                else:
                    control_for_obs = obs    
                
                K[obs][state] = control_simple(xped, state, control_for_obs, Ncar)                
    return K


if __name__ == "__main__":
    xped = 7
    Ncar = 11
    xcar = 1
    vcar = 2
    obs = "ped"
    print("Position: ", str(xcar)," and speed: ", str(vcar))

    car_stopped = False
    while not car_stopped:
        new_state = control_simple(xped, (xcar, vcar), obs, Ncar)
        xcar, vcar = new_state
        if xcar == Ncar or vcar == 0:
            car_stopped = True
        print("Position: ", str(new_state[0])," and speed: ", str(new_state[1]))
    
    Vhigh = 4
    env_obs = ["ped", "obs", "empty"]
    K = control_dict(Ncar, Vhigh, env_obs, xped)
    st()
