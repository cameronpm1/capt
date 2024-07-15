import numpy as np
import scipy
from typing import Any, Dict, Optional
from matplotlib import pyplot as plt

from astropy import units as u

from dynamics.base_dynamics import baseDynamics
from util.util import *


class twodDynamics(baseDynamics):

    def __init__(
        self,
        timestep: int = 1,
        horizon: int = 10,
        pos: list[float] = np.array([0.0, 0.0]), #initial position w/ respect to orbit
        vel: list[float] = np.array([0.0, 0.0]), #initial velocity w/ respect to orbit
        euler: list[float] = np.array([0.0]), #initial euler angles
        omega: list[float] = np.array([0.0]), #angular velocity vector
        data: Optional[Dict[str, Any]] = None,
        max_control: list[float] = [0.3,0.3,0.3,0.3],
    ):
        
        super().__init__(
            timestep=timestep,
            horizon=horizon,
            pos=pos,
            vel=vel,
            euler=euler,
            omega=omega,
        )

        self.mass = data['mass']
        self.inertia = data['inertia']
        self.max_control = np.array(max_control)

        self.initialize_state()
        self.initialize_control()
        self.initialize_state_matrix()
        self.initialize_control_matrix()


    '''
    MPC SUPPORT FUNCTIONS

    A,B,C,D,Q,R are all solely to be used by the Linear-Quadratic MPC
    A and B are the control and state matrices for the simplified
    linear dyanmics of a quadcopter (from same paper). 

    convert_input transforms the motor command speeds to
    forces and torques using support fuctions from the 
    optomizer package used in the MPC

    MPC equations are presented as:

    dx/dt = Ax + Bu
    y = Cx + Du
    cost = xQx + uDu

    They assume ONLY small deviations in euler angles from 0 (~5deg)
    '''

    @property
    def A(self):
        # Linear state transition matrix
        A = np.zeros((self.state.size, self.state.size))
        A[0][2] = 1
        A[1][3] = 1
        A[4][5] = 1
        return A
    
    @property
    def B(self):
        # Control matrix
        B = np.zeros((self.state.size,self.control.size))
        B[2][0] = 1/self.mass
        B[3][1] = 1/self.mass
        B[5][2] = 1/self.inertia
        return B
    
    @property
    def C(self):
        C = np.eye(self.state.size)
        return C

    @property
    def D(self):
        D = np.zeros((self.state.size, self.control.size))
        return D
    
    @property
    def Q(self):
        # State cost
        Q = np.eye(self.state.size)
        Q[0, 0] = 10.  # x pos
        Q[1, 1] = 10.  # y pos
        Q[2, 2] = 10.  # x vel
        Q[3, 3] = 10.  # y vel
        Q[4, 4] = 10.  # orientation
        return Q

    @property
    def R(self):
        # Actuator cost
        R = np.eye(self.control.size)*.1
        R[4, 4] = 0.001
        return R

    '''
    END OF MPC SUPPORT FUNCTIONS
    '''
        
         
    def reset_state(self) -> None:
        super().reset_state()
        self.initialize_state()

    def initialize_control(self) -> None:
        self.control = np.zeros((3,1))

    def initialize_state_matrix(self) -> None:
        super().initialize_state_matrix()
        self.state_matrix[0][2] = 1
        self.state_matrix[1][3] = 1
        self.state_matrix[4][5] = 1

    def initialize_control_matrix(self) -> None:
        self.control_matrix = np.zeros((self.state.size,self.control.size))
        self.control_matrix[2][0] = 1/self.mass
        self.control_matrix[3][1] = 1/self.mass
        self.control_matrix[5][2] = 1/self.inertia
           

    def set_control(self, control) -> None:
        control = np.clip(control, -self.max_control, self.max_control)
        for i in range(self.control.size):
            self.control[i] = control[i]

    def compute_derivatives(self, state, t) -> list[float]:

        dxdt = np.matmul(self.state_matrix,state) + np.squeeze(np.matmul(self.control_matrix,self.control))
        return dxdt
    
    def forward_step(self) -> list[float]:
        timerange = np.arange(self.time, self.time+(self.timestep*self.horizon), self.timestep)
        sol = scipy.integrate.odeint(
            self.compute_derivatives,
            self.state,
            timerange,
        )
        self.time += self.timestep*self.horizon
        self.state = sol[-1]
        return sol