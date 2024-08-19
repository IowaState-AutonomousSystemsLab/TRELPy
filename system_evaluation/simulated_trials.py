### Simulated Car Trials
# Driving down a straight line
# Apurva Badithela

import numpy as np
import random
from pdf import set_trace as st
from controllers.simple_controller import control_dict, prop_control_dict
import control as ct

#################################################
# Car Dynamics. Car state: [x, y, theta]
# KinCar: Simple Car dynamics. Car driving in straight line along Y-axis and zero-order hold controller. 
# DiffFlatCar: Implementing differentially flat controllers from the Python control toolbox.
class KinCar:
    def __init__(self, x_init, u_init, L=3):
        self.x = x_init
        self.u_init = u_init
        self.ydot = self.unit[0]*np.sin(self.x[2])
        self.L = L # Wheel base
        pass

    def vehicle_update(self, us, uphi):
        dx = np.array([np.cos(self.x[2]) * us, np.sin(self.x[2]) * us, (us/self.L) * np.tan(uphi)])
        return dx

    def simulate_step(self, us, uphi, dt):
        dx = self.vehicle_update(us, uphi)
        self.x += dx*dt       

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
c2d = 1.0/DIS_TO_CONT # Needs to be further approximated to get into integer form.
dt = 0.1 # seconds. Each individual update step
x0 = np.array([0, 0, np.pi/2]) # Driving up
u0 = np.array([1, 0])
image_rate = 0.1 # Hz.
Ntrials = 100

#################################################
# Sample observation according to confusion matrix
def sample(C, ped):

    return obs


#################################################
# Simulate trials
# Each trial is a random run of the car starting from origin.
def trial():
    car = KinCar(x0, u0)
    traj = [(car.x0, car.ydot)]
    while car.ydot > 0:
        


    return traj

