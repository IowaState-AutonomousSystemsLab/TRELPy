# Controller for empty observation:
def emptyK(N, xcar, vcar, Vlow, Vhigh, xped):
    sys_vars = {}
    sys_vars['xcar'] = (1, N)
    sys_vars['vcar'] = (Vlow, Vhigh)
    env_vars = {}
    env_vars['xempty'] = (0,1) # Pavement is empty

    sys_init = {'xcar='+str(xcar), 'vcar='+str(vcar)}
    env_init = {'xempty='+str(1)}
     # Test lines:
    sys_init = {'xcar='+str(xcar), 'vcar='+str(vcar)}
    env_init = {'xempty='+str(1)}

    sys_prog = set() # For now, no need to have progress
    env_prog = set()

    sys_safe = set()

    # Env safe spec: If env is empty, it always remains empty
    env_spec = {'xempty=1 -> X(xempty=1)'}
    env_safe = set()
    env_safe |= env_spec

    # Environment safety specs: Static pedestrian
    # env_safe |= {'xped='+str(xped)+'-> X(xped='+str(xped)+')'}
    # Spec: If you don't see an object, keep moving:
    spec_k = {'(xempty=1 && vcar=0)->X(vcar=1)'}
    sys_safe |= spec_k
    for vi in range(1, Vhigh):
        spec_k = {'(xempty=1 && vcar='+str(vi)+')->X(vcar='+str(vi+1)+')'}
        sys_safe |= spec_k
    spec_k = {'(xempty=1 && vcar='+str(Vhigh)+')->X(vcar='+str(Vhigh)+')'}
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