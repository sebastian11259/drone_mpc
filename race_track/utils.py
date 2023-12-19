from pathlib import Path
import csv
from typing import Union, Dict, Tuple, Any
from collections import deque

import pybullet as p
import numpy as np
import yaml

def readYAML(config_file: Union[Path, str]) -> Any:
    with open(config_file) as f:
        config=yaml.load(f)
    return config

def readCSV(track_file: Union[Path, str]):
    with open(track_file) as f:
        reader=csv.reader(f)
        data=list(reader)
    return data

def calculateRelativeObseration(obj1: np.array , obj2: np.array) -> np.array:
    """
    obj = ([x, y, z], [qx, qy, qz, qw])
    """
    # TODO - keep info about quaterions and not shperical
    pos1, ort1 = obj1
    pos2, ort2 = obj2

    # Step 1 - Vector between two points
    vec_diff = pos2 - pos1
    quat_diff = p.getDifferenceQuaternion(ort1, ort2)
    inv_p, inv_o = p.invertTransform([0,0,0], ort1)
    rot_vec, _ = p.multiplyTransforms(inv_p, inv_o,
                               vec_diff, [0, 0, 0, 1])
    # Step 2 - calculate shperical coordinates
    r, theta, phi = cart2shp(rot_vec)
    # Step 3 - calculate angle between normals
    _, alpha = p.getAxisAngleFromQuaternion(quat_diff)
    return np.array([r, theta, phi, alpha])


def cart2shp(cart: np.array) -> Tuple[float, float, float]:
    xy = np.sqrt(cart[0]**2 + cart[1]**2)
    r = np.sqrt(xy**2 + cart[2]**2)
    theta = np.arctan2(cart[1], cart[0]) # for elevation angle defined from Z-axis down
    phi = np.arctan2(xy, cart[2])
    return r, theta, phi
