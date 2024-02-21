from __future__ import print_function

from framework.control.tulip_controller import design_K
from pdb import set_trace as st
from empty_spec import empty_K
from pedestrian_spec import pedestrian_K
from obstacle_spec import obstacle_K

# Construct controller for each observation:
def construct_controllers(N, Vlow, Vhigh, xped, vcar, xcar = 1, control_dir=None):    
    # When a pedestrian is observed:
    env_vars, sys_vars, env_init, sys_init, env_safe, sys_safe, env_prog, sys_prog = pedestrian_K(N, xcar, vcar, Vlow, Vhigh, xped)
    # pretty_print_specs(sys_safe, "sys safe")
    Kped = design_C(env_vars, sys_vars, env_init, sys_init, env_safe, sys_safe, env_prog, sys_prog)
    if control_dir:
        ctrl_path = os.path.join(control_dir, "ped_controller.py")
        write_python_case(ctrl_path, Kped)

    # When something other than a pedestrian is observed:
    env_vars, sys_vars, env_init, sys_init, env_safe, sys_safe, env_prog, sys_prog = obstacle_K(N, xcar, vcar, Vlow, Vhigh, xped)
    Knot_ped = design_C(env_vars, sys_vars, env_init, sys_init, env_safe, sys_safe, env_prog, sys_prog)
    if control_dir:
        ctrl_path = os.path.join(control_dir, "not_ped_controller.py")
        write_python_case(ctrl_path, Knot_ped)

    # When nothing is observed:
    env_vars, sys_vars, env_init, sys_init, env_safe, sys_safe, env_prog, sys_prog = empty_K(N, xcar, vcar, Vlow, Vhigh, xped)
    Kempty = design_C(env_vars, sys_vars, env_init, sys_init, env_safe, sys_safe, env_prog, sys_prog)
    if control_dir:
        ctrl_path = os.path.join(control_dir, "empty_controller.py")
        write_python_case(ctrl_path, Kempty)

    K = dict()
    K["ped"] = Kped
    K["obj"] = Knot_ped
    K["empty"] = Kempty

    return K

if __name__=='__main__':
    # Simple example of pedestrian crossing street:
    N = 5
    Vhigh = 1
    Vlow = 0
    xcar = 1
    vcar = Vhigh
    xped = 3

    # When a pedestrian is observed:
    env_vars, sys_vars, env_init, sys_init, env_safe, sys_safe, env_prog, sys_prog = pedestrianK(N, xcar, vcar, Vlow, Vhigh, xped)
    Kped = design_C(env_vars, sys_vars, env_init, sys_init, env_safe, sys_safe, env_prog, sys_prog)
    write_python_case("ped_controller.py", Kped)

    # When something other than a pedestrian is observed:
    env_vars, sys_vars, env_init, sys_init, env_safe, sys_safe, env_prog, sys_prog = not_pedestrianK(N, xcar, vcar, Vlow, Vhigh, xped)
    Knot_ped = design_C(env_vars, sys_vars, env_init, sys_init, env_safe, sys_safe, env_prog, sys_prog)
    write_python_case("not_ped_controller.py", Knot_ped)

    # When nothing is observed:
    env_vars, sys_vars, env_init, sys_init, env_safe, sys_safe, env_prog, sys_prog = emptyK(N, xcar, vcar, Vlow, Vhigh, xped)
    Kempty = design_C(env_vars, sys_vars, env_init, sys_init, env_safe, sys_safe, env_prog, sys_prog)
    write_python_case("empty_controller.py", Kempty)