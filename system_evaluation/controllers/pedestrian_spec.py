# Controller for not_pedestrian observation:
def pedestrianK(N, xcar, vcar, Vlow, Vhigh, xped):
    sys_vars = {}
    sys_vars['xcar'] = (1, N)
    sys_vars['vcar'] = (Vlow, Vhigh)
    env_vars = {}
    env_vars['xped'] = (0,1) # Pedestrian is present or absent

    sys_init = {'xcar='+str(xcar), 'vcar='+str(vcar)}
    env_init = {'xped='+str(xped)}

    # Test lines:
    # sys_init = {'xcar='+str(xcar)}
    env_init = set()
    # ========================= #
    sys_prog = set() # For now, no need to have progress
    env_prog = set()
    xcar_jj = (xped-1) # eventual goal location for car
    #sys_prog = set({'xcar = '+str(xcar_jj)})

    sys_safe = set()
    env_safe = set()

    # Environment safety specs: Static pedestrian
    for xi in range(0, 2):
        env_safe |= {'xped='+str(xi)+'-> X(xped='+str(xi)+')'}

    # Safety specifications to specify that car must stop before pedestrian:
    sys_safe |= {'((xped = 1) ||!(xcar = '+str(xcar_jj)+' && vcar = 0))'}
    car_states = ""
    for xcar_ii in range(xcar_jj, N+1):
        if car_states == "":
            car_states = "xcar = "+str(xcar_ii)
        else:
            car_states = car_states + " || xcar = " + str(xcar_ii)
    sys_safe |= {'(!(xped = 1)||!('+car_states+')||(vcar = 0 && xcar = '+str(xcar_jj)+'))'}

    # Safety specs for car to not stop before car reaching pedestrian sidewalk
    for xi in range(1, xcar_jj):
        sys_safe |= {'!(xcar = '+str(xi)+' && vcar = 0)'}

    # Add system dynamics to safety specs:
    for ii in range(1, N+1):
        for vi in range(Vlow, Vhigh+1):
            if vi==0:
                spec_ii = {'((xcar='+str(ii)+') && (vcar=0))-> X((vcar=0 || vcar=1) && xcar='+str(ii)+')'}
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