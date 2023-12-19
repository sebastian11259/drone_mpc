from typing import List, Tuple

import pybullet as p
import numpy as np

from .gate import Gate
from .utils import calculateRelativeObseration

class Segment:
    def __init__(
            self,
            start_pos: np.array,
            gate: Gate,
            ):
        # Elements related to gates and their relative pos
        self.start_pos=start_pos
        self.gate=gate
        self.segment=gate.pos-start_pos
        self.norm_segment=np.linalg.norm(self.segment)

    def projectPoint(self, pos: np.array) -> np.array:
        projection=np.dot(pos-self.gate.pos, self.segment)
        return projection/self.norm_segment
    
    def getRelativeObs(self, d_obj: Tuple[np.array, np.array]) -> np.array:
        g_obj=self.gate.pos, self.gate.quat
        return calculateRelativeObseration(d_obj, g_obj)        

    def startPosition(self):
        pos=self.start_pos+self.segment/2
        g_dir=self.segment/self.norm_segment
        # Cast to Z axis
        z_cast=np.dot(g_dir, np.array([0,0,1]))
        # Calculate projecion on XY plane
        xy_cast=g_dir-np.array([0,0,z_cast])
        oz_angel=np.arctan2(xy_cast[1], xy_cast[0])
        ort=p.getQuaternionFromEuler([0, 0, oz_angel])
        return pos, ort

    def segmentFinished(self, d_pos: np.array) -> bool:
        # Calculate distance to the plane defined by the gate'
        diff_vec=self.gate.pos-d_pos
        t_pos, _ = p.invertTransform(
            position=diff_vec,
            orientation=self.gate.quat,
        )
        return t_pos[0] > 0

    def distanceToSegment(self, d_pos: np.array) -> float:
        # Equations as in https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        p = d_pos
        a = self.start_pos
        n = self.segment/self.norm_segment

        distance = np.linalg.norm((p-a) - np.dot((p-a), n)*n)
        return distance