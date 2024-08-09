import numpy as np
import copy
import pyvista as pv
from typing import Any, Dict, Type, Optional, Union

from dynamics.sat_dynamics import satelliteDynamics
from dynamics.base_dynamics import baseDynamics
from dynamics.static_object import staticObject
from util.util import *

class dynamicObject(staticObject):

    def __init__(
            self,
            dynamics: Type[baseDynamics],
            mesh: Union[Dict[str, list[Any]],Type[pv.DataSet]],
            name: Optional[str] = None,
            pos: list[float] = [0,0,0],
            dim: int = 3,
    ):

        super().__init__(
            mesh = mesh,
            name = name,
            pos=pos,
            dim=dim,
        )

        self.dynamics = dynamics
        self.dim = self.dynamics.dim
    
    def update_points(self) -> None:
        #return spacecraft mesh points translated to current location using euler angles
        if self.vista:
            dcm = dcm_zyx(self.dynamics.get_euler())
            current_pos = self.dynamics.get_pos()
            for i,point in enumerate(self.mesh.points):
                new_point = np.matmul(dcm,point) + current_pos
                self.temp_mesh.points[i] = new_point
        else:
            if self.dim == 2:
                dcm = dcm_xy(self.dynamics.get_euler())
            else:
                dcm = dcm_zyx(self.dynamics.get_euler())
            current_pos = self.dynamics.get_pos()
            for i,point in enumerate(self.mesh['points']):
                new_point = np.matmul(dcm,point) + current_pos
                self.temp_mesh['points'][i] = new_point

    
    def step(self) -> None:
        states = self.dynamics.forward_step()
        self.update_points()

    def reset(self) -> None:
        self.dynamics.reset_state()
        self.update_points()

    def get_state(self) -> list[float]:
        return self.dynamics.get_state()