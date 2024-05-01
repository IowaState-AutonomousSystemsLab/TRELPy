import numpy as np
import pdb 

def get_state(x, v, Vhigh, Vlow=0):
    state_num = int((Vhigh-Vlow+1)*(x-1)+v)
    state_str = "S"+str(state_num)
    return state_num, state_str

def get_formula_states_ev_stop(xcar_stop, Vhigh, Vlow=0):
        bst = set()
        _, good_state_str = get_state(xcar_stop, 0, Vhigh)
        gst = {good_state_str}
        for vi in range(0,Vhigh+1):
            _, state_str = get_state(xcar_stop, vi, Vhigh)
            if state_str not in gst:
                bst |= {state_str}
        
        bad = "" # Expression for bad states
        good = "" # Expression for good states
        for st in list(gst):
            if good == "":
                good = good + "\"" + st+"\""
            else:
                good = good + "|\""+st+"\""
        for st in list(bst):
            if bad == "":
                bad = bad + "\"" + st+"\""
            else:
                bad = bad + "|\""+st+"\""
        return good, bad, gst, bst

def set_crosswalk_cell(Ncar, xmax_stop):
    '''
    Inputs: 
    Vhigh: Highest possible speed for a car
    Ncar: Number of discrete cells for a car
    xmax_stop: The maximum cell that a car can stop by at its highest initial speed. 
    Vhigh need not be the car's highest initial speed.
    '''
    
    Nped = Ncar - 1 # Penultimate cell
    
    # xped = np.random.randint(xmax_stop+1, Ncar) # Pick a pedestrian cell from [xmax_stop+1, Ncar]
    xped = xmax_stop+1
    xcar_stop = xped - 1 # Cell state of the car by which v = 0
    assert(xcar_stop >= xmax_stop)
    return int(xped), int(xcar_stop)

def formula_deprec(xcar_stop, Vhigh, Vlow=0):
    '''
    Formula to compute the probability of not reaching a bad state until a good state has been visited.
    This formula is not used in the results but was kept in the code.
    '''
    bad_states = set()
    good_state = set()
    
    good, bad, gst, bst = get_formula_states_ev_stop(xcar_stop, Vhigh, Vlow=0)
    good_state |= gst
    bad_states |= bst
    formula = "P=?[!("+str(bad)+") U "+str(good)+"]"
    return formula

def formula_visit_bad_states(Ncar, xcar_stop, Vhigh, Vlow=0):
    '''
    Probability of never visiting a good state, and never ...
    '''
    bad_states = set()
    good_state = set()
    
    good, bad, gst, bst = get_formula_states_ev_stop(xcar_stop, Vhigh, Vlow=0)

    phi1 = "!("+good+")"
    phi2 = "("+good+") | !("+bad

    for xcar_ii in range(xcar_stop+1, Ncar+1):
        good, bad, gst, bst = get_formula_states_ev_stop(xcar_ii) # We only want the bad states; ignore the good states output here
        bad_states |= bst
        phi2 = phi2 + "|" + bad
    phi2 = phi2 + ")"
    formula = "P=?[G("+str(phi1)+") && G("+str(phi2)+")]"
    return formula

def formula_ev_good(xcar_stop, Vhigh, Vlow=0):
    '''
    Probability of eventually reaching a good state
    '''
    good_state = set()
    
    good, bad, gst, bst = get_formula_states_ev_stop(xcar_stop, Vhigh, Vlow=0)
    good_state |= gst
    good_st = '|'.join(list(good_state))
    formula = 'P=?[F(\"'+good_st+'\")]'
    return formula

def formula_not_stop(xcar_stop, Vhigh, N):
    '''
    Not stop until xcar_stop
    '''
    speed_cells = ""
    for vi in range(1,Vhigh+1):
        _, state_str = get_state(xcar_stop, vi, Vhigh)
        if speed_cells == "":
            speed_cells = speed_cells + "\"" + state_str+"\""
        else:
            speed_cells = speed_cells + "|\""+state_str+"\""
    
    for xi in range(xcar_stop+1, N+1):
        for vi in range(1,Vhigh+1):
            _, state_str = get_state(xi, vi, Vhigh)
            speed_cells = speed_cells + "|\""+state_str+"\""
    
    # probability of reaching crosswalk at nonzero speed
    # this will be the same as doing the right action because if the car stops beforehand, it stops permanently and
    # would never reach this state.
    # But, we cannot just evaluate it at this state. The car must continue
    # driving; due to its speed; it might not even register this state. 
    formula = 'P=?[F('+speed_cells+')]' 
    return formula