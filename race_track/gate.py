from pathlib import Path
from typing import Callable, Union

import pybullet as p
import numpy as np

class Gate:
    def __init__(
            self,
            pos: np.array,
            quat: np.array,
            scale: float,
            asset: Path,
            clientID: Union[int, None],
            ):
        # Save position and calculate normal
        self.pos=pos
        self.quat=quat
        self.scale=scale
        # Load asset to pybullet client
        if not (clientID is None): 
            self.load_bullet=self.loadBullet(asset, clientID)
            self.load_bullet()

    def loadBullet(self, asset: Path, clientID: int) -> Callable:
        # import pdb; pdb.set_trace()
        def func():
            urdf_id=p.loadURDF(
                    str(asset),
                    self.pos,
                    self.quat,
                    globalScaling=self.scale,
                    useFixedBase=1,
                    physicsClientId=clientID,
            )
            self.urdf_id=urdf_id
        return func

    def field_reward(self, d_pos: np.array) -> float:
        diff_vec = d_pos-self.pos 
        # Transform drone position to gate reference frame
        t_pos, _ = p.invertTransform(position=diff_vec,
                                     orientation=self.quat)

        dp, dn = t_pos[0], np.sqrt(t_pos[1]**2 + t_pos[2]**2)
        # Normalize dn
        dn /= self.scale
        # TODO - check the cooeficent of 1.5
        f = lambda x: max(1-x/1.5, 0.0)
        v = lambda x, y: max((1- y) * (x/6.0), 0.05)
        filed_reward = -f(dp)**2 * (1 - np.exp(- 0.5 * dn**2 / v(1.0, f(dp))))
        return filed_reward
    
    
