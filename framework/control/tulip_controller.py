from __future__ import print_function

import logging
from tulip import spec, synth, hybrid, transys

# This script invokes TuLiP to construct a controller for the system with respect to a
# temporal logic specification based on the observed outputs of the perception algorithm
# Inputs to the controller synthesis function are: discrete_dynamics (disc_dynamics),
# cell/set of env. variables (env_vars), cell/set of system variables (sys_vars),
# cell/set of initial env. variables (env_init), cell/set of initial system variables (sys_init),
# enviornment and system safety and progress specifications: env_safe/env_prog and sys_safe/sys_prog.

def design_K(env_vars, sys_vars, env_init, sys_init, env_safe, sys_safe, env_prog, sys_prog):
    logging.basicConfig(level=logging.WARNING)
    show = False

    # Constructing GR1spec from environment and systems specifications:
    specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                        env_safe, sys_safe, env_prog, sys_prog)
    specs.moore = True
    specs.qinit = '\E \A'

    # Synthesize
    ctrl = synth.synthesize(specs)
    try:
        assert ctrl is not None, 'unrealizable'
    except:
        st()

    return ctrl

def pretty_print_specs(spec_set, spec_name):
    print("=============================================")
    print(spec_name)
    for spec in spec_set:
        print(spec)
