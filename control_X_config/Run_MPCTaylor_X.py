import pybullet as p
import numpy as np
import time

from env.MPCAviary import MPCAviary
from control_X_config.MPC_Taylor_X import MPC


from drone_sim.gym_pybullet_drones.utils.enums import DroneModel


def run_mpc(track_path: str, mpc_file: str, gui: bool = True):
    _T = 0.25
    _dt = 0.01
    _N = _T / _dt
    pyb_freq = 250
    ctrl_freq = 50

    mpc = MPC(_T, _dt, mpc_file)
    env = MPCAviary(
        gui = gui,
        pyb_freq = pyb_freq,
        ctrl_freq = ctrl_freq,
        track_path=track_path,
        drone_model=DroneModel.CF2X,
        kt=1000
    )

    infos = []
    # env.step_counter

    compute_time = 0

    quad_act = [14468.4, 14468.4, 14468.4, 14468.4]

    obs, _ = env.reset()
    t = 0
    while True:
        # t = env.step_counter / pyb_freq
        ref_traj = env.planer.getRefPath(t, _dt, _T)
        t += 1/ctrl_freq

        obs = obs.tolist()

        # print("OBS: ", obs[0:3])
        # print("REF: ", ref_traj[0:3])
        # print("------------------------------------------------------------------------------------")

        # print(obs)
        # print(ref_traj[0:13])

        ref_traj = obs + ref_traj + list(quad_act)



        start = time.time()
        quad_act, pred_traj = mpc.solve(ref_traj)
        compute_time += time.time() - start



        obs, reward, terminated, truncated, info = env.step(quad_act)

        info = {
            "quad_obs": obs,
            "quad_act": quad_act,
            "pred_traj": pred_traj,
            "ref_traj": ref_traj
        }
        infos.append(info)

        if terminated or truncated:
            break

    return infos, t, env.planer.getTime(1), compute_time






if __name__ == "__main__":
    mpc_file = "mpc.so"
    # track_path = "../assets/tracks/thesis-tracks/straight_track.csv"
    track_path = "../assets/tracks/circle_track.csv"
    # track_path = "../assets/tracks/thesis-tracks/dive_track.csv"

    gui = True

    i, t, t_, ct = run_mpc(track_path, mpc_file, gui)

    print('Czas przelotu:   ', t)
    print('Czas planera:   ', t_)
    print('Czas obliczeń: ', ct)
