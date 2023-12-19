from pathlib import Path
from typing import List, Union, Tuple

import pybullet as p
import numpy as np

from .gate import Gate
from .segment import Segment
from .utils import readCSV 

class Track:
    def __init__(
           self,
           track_path: Union[Path, str],
           asset_path: Union[Path, str, None]=None,
           clientID: Union[int, None]=None,
           ):
        self.asset_path=Path(asset_path)
        self.track_data=readCSV(track_path)
        print(self.track_data)
        self.gates=self._createGates(clientID)
        self.segments=self._createSegments()

    def _createGates(self, clientID: int) -> List[Gate]:
        gates=[]
        for gate_data in self.track_data[1:]:
            asset=self.asset_path / gate_data[0]
            pos=np.array([
                gate_data[1],
                gate_data[2],
                gate_data[3]]
            ).astype(np.float64)
            quat=np.array([
                gate_data[4],
                gate_data[5],
                gate_data[6],
                gate_data[7]]
            ).astype(np.float64)
            scale=float(gate_data[8])
            gate=Gate(pos, quat, scale, asset, clientID)
            gates.append(gate)
        return gates
            
    def _createSegments(self) -> List[Segment]:
        start_data=self.track_data[0]
        pos=np.array([start_data[0],start_data[1],start_data[2]]).astype(np.float64)
        #quat=np.array([start_data[4],start_data[5],start_data[6],start_data[7]]).astype(np.float64)
        segments=[]
        for gate in self.gates:
            segment=Segment(pos, gate) 
            pos=gate.pos
            segments.append(segment)
        return segments

    def getTrackStart(self) -> Tuple[np.array]:
        start_data=self.track_data[0]
        pos=np.array([start_data[0],start_data[1],start_data[2]]).astype(np.float64)
        quat=np.array([start_data[3],start_data[4],start_data[5],start_data[6]]).astype(np.float64)
        return pos, quat

    def getEndPoint(self) -> Tuple[np.array]:
        """
        Append end point 2 meters behind last gate for the geometric planer
        """
        last_gate=self.gates[-1]
        pos, quat = last_gate.pos, last_gate.quat
        diff=[-2, 0, 0] # Vector of 2 meters in x direcion
        d_pos, _ = p.invertTransform(diff, [quat[0], quat[1], -quat[2], quat[3]])
        e_pos = pos + np.array(d_pos) 
        return e_pos, quat


    def reloadGates(self):
        # Reload gate to the enviroment aster bullet reset
        for gate in self.gates:
            gate.load_bullet()
