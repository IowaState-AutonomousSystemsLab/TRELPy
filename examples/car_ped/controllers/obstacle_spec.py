# Function constructing transition system, specification variables for pedestrian/car example:
# Takes as input the geometry of the sidewalk:
# N: No. of cells of the car, Nped: No. of cells of the crosswalk, xped: initial cell number of pedestrian, xcar: initial cell of car, vcar: initial velocity of car, xcross_start: Cell number of road at which the crosswalk starts
# Vlow: Lower integer speed bound for car, Vhigh: Upper integer speed bound for car
def obstacle_K(N, xcar, vcar, Vlow, Vhigh, xped):
    sys_vars = {}
    sys_vars['xcar'] = (1, N)
    sys_vars['vcar'] = (Vlow, Vhigh)
    env_vars = {}
    env_vars['xobj'] = (0,1) # Difficult to have a set of just 1

    sys_init = {'xcar='+str(xcar), 'vcar='+str(vcar)}
    env_init = {'xobj='+str(1)}

     # Test lines:
    sys_init = {'xcar='+str(xcar), 'vcar='+str(vcar)}
    env_init = {'xobj='+str(1)}

    sys_prog = set() # For now, no need to have progress
    env_prog = set()

    sys_safe = set()
    env_safe = {'xobj=1 -> X(xobj=1)'}

    # Object controllers: If you see an object that is not a pedestrian, then keep moving:
    # spec_k = {'(xobj=1)->X(vcar=1)'}
    # sys_safe |= spec_k

    for vi in range(Vhigh, 1, -1):
        spec_k = {'(xobj=1 && vcar='+str(vi)+')->X(vcar='+str(vi-1)+')'}
        sys_safe |= spec_k
    spec_k = {'(xobj=1 && vcar=1) -> X(vcar=1)'}
    sys_safe |= spec_k
    # Add system dynamics to safety specs:
    for ii in range(1, N+1):
        for vi in range(Vlow, Vhigh+1):
            if vi==0:
                spec_ii = {'((xcar='+str(ii)+') && (vcar=0))-> X((vcar=1) && xcar='+str(ii)+')'}
                sys_safe|=spec_ii
            elif vi == Vhigh:
                xf_ii = min(ii+vi, N)
                spec_ii = {'((xcar='+str(ii)+') && (vcar='+str(vi)+'))-> X((vcar='+str(vi)+'|| vcar='+str(vi-1)+') && xcar='+str(xf_ii)+')'}
                sys_safe|=spec_ii
            else:
                xf_ii = min(ii+vi, N)
                spec_ii = {'((xcar='+str(ii)+') && (vcar='+str(vi)+'))-> X((vcar='+str(vi)+'|| vcar='+str(vi-1)+'|| vcar='+str(vi+1)+') && xcar='+str(xf_ii)+')'}
                sys_safe|=spec_ii
    return env_vars, sys_vars, env_init, sys_init, env_safe, sys_safe, env_prog, sys_prog