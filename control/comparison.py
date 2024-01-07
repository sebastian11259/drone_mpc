from env.MPCAviary import MPCAviary
from control.MPC import MPC as MPC
from control.MPC_Taylor import MPC as MPC_Taylor
from control.qLPV_MPC import MPC as qLPV_MPC
from drone_sim.gym_pybullet_drones.utils.enums import DroneModel

import time


def run_mpc(mpc_ver: int, track_path: str, _T: float, _dt: float, divisor: int, kt: int, gui: bool = False):

    N = _T/_dt
    pyb_freq = 250
    ctrl_freq = 50

    if (mpc_ver == 0):
        mpc = MPC(_T, _dt, divisor)
    elif (mpc_ver == 1):
        mpc = MPC_Taylor(_T, _dt, divisor)
    else:
        mpc = qLPV_MPC(_T, _dt, divisor)

    env = MPCAviary(
        gui=gui,
        pyb_freq=pyb_freq,
        ctrl_freq=ctrl_freq,
        track_path=track_path,
        drone_model=DroneModel.CF2P,
        kt=kt,
        wind=False
    )

    full_compute_time = 0
    avg_compute_time = 0

    quad_act = [14468.4, 14468.4, 14468.4, 14468.4]

    obs, _ = env.reset()
    t = 0
    while True:
        ref_traj = env.planer.getRefPath(t, _dt, _T)
        t += 1 / ctrl_freq

        obs = obs.tolist()

        ref_traj = obs + ref_traj

        if (mpc_ver == 1):
            ref_traj += list(quad_act)
        elif (mpc_ver == 2):
            ref_traj += [0 for _ in range(13)]

        quad_act, pred_traj = mpc.solve(ref_traj)

        obs, reward, terminated, truncated, info = env.step(quad_act)

        if terminated or truncated:
            env.close()
            break


if __name__ == "__main__":
    track_path = "../assets/tracks/circle_track.csv"

    for i in range(3):
        run_mpc(i, track_path, 0.5, 0.02, 4, 1000, True)