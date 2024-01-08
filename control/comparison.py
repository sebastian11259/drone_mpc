from env.MPCAviary import MPCAviary
from control.MPC import MPC as MPC
from control.MPC_Taylor import MPC as MPC_Taylor
from control.qLPV_MPC import MPC as qLPV_MPC
from drone_sim.gym_pybullet_drones.utils.enums import DroneModel

import numpy as np
import pandas as pd
import time

from path_planing import convert_for_planer, PathPlanner


def run_mpc(mpc_ver: int, track_path: str, _T: float, _dt: float, divisor: int, kt: int, planner: PathPlanner, gui: bool = False, ):

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
        wind=False,
        initialize_planner=False
    )

    compute_times = []


    quad_act = [14468.4, 14468.4, 14468.4, 14468.4]

    obs, _ = env.reset()
    t = 0


    flight_traj = []

    while True:
        flight_traj.append(obs)

        ref_traj = planer.getRefPath(t, _dt, _T)
        t += 1 / ctrl_freq

        obs = obs.tolist()

        ref_traj = obs + ref_traj

        if (mpc_ver == 1):
            ref_traj += list(quad_act)
        elif (mpc_ver == 2):
            ref_traj += [0 for _ in range(13)]

        start = time.time()
        quad_act, pred_traj = mpc.solve(ref_traj)
        compute_times.append(time.time() - start)

        obs, reward, terminated, truncated, info = env.step(quad_act)

        if terminated or truncated:
            env.close()
            break

    info = {
        "planner_time": planer.getTime(1),
        "flight_time": t,
        "compute_times": compute_times,
        "success": terminated,
        "flight_traj": flight_traj,
        "ref_traj": ref_traj
    }

    return info

if __name__ == "__main__":
    track_paths = [
        "../assets/tracks/circle_track.csv"
    ]

    clear_df = pd.DataFrame(columns=[
        "mpc_ver",
        "track",
        "success",
        "horizon",
        "sampling_time",
        "divisor",
        "kt",
        "flight_time",
        "planner_time",
        "total_time",
        "avg_time",
        "flight_traj",
        "ref_traj"])

    _kt = [1000, 2000, 5000, 7000, 10000, 12000, 15000, 20000, 30000, 50000]
    T_dt = [
        (0.2, [0.005, 0.01, 0.02]),
        (0.25, [0.005, 0.01, 0.02, 0.05]),
        (0.5, [0.005, 0.01, 0.02, 0.025, 0.05, 0.1]),
        (0.75, [0.005, 0.01, 0.025, 0.05, 0.075]),
        (1, [0.025, 0.05, 0.1, 0.2])
    ]


    divisor = 4


    df = clear_df
    df.to_csv('Results/wyniki.csv', index=False)
    for ver in range(1,3):
        for track_path in track_paths:
            for kt in _kt:
                points = convert_for_planer(track_path)
                planer = PathPlanner(np.array(list(map(lambda x: x[0], points))), max_velocity=20, kt=kt)

                for T, _dt in T_dt:
                    for dt in _dt:

                        info = run_mpc(ver, track_path, T, dt, divisor, kt, False)

                        df = df._append({
                            "mpc_ver": ver,
                            "track": "t",
                            "success": info['success'],
                            "horizon": T,
                            "sampling_time": dt,
                            "divisor": divisor,
                            "kt" : kt,
                            "flight_time": info['flight_time'],
                            "planner_time": info['planner_time'],
                            "total_time": sum(info['compute_times']),
                            "avg_time": sum(info['compute_times']) / len(info['compute_times']),
                            "flight_traj": info['flight_traj'],
                            "ref_traj": info['ref_traj']
                            }, ignore_index=True)

                df.to_csv('Results/wyniki.csv', mode='a', header=False, index=False)
                df = clear_df
