import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from pathlib import Path

from race_track import Track


def convert_for_planer(track_path: Path):
    track = Track(track_path, "assets")
    
    T = [track.getTrackStart()]
    for gate in track.gates:
        T.append((gate.pos, gate.quat))

    T.append(track.getEndPoint()) 

    return T

def visualize_points(T):
    # Sample list of 3D points (replace with your own data)
    points = list(map(lambda x: x[0], T))

    # Extract x, y, and z coordinates from the list of points
    x_coords, y_coords, z_coords = zip(*points)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points as scatter
    ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o')
    ax.plot(x_coords, y_coords, z_coords, c='r', linestyle='-')

    # Set labels for the axes
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    # Show the 3D plot

    return fig, ax

def check_if_trajectory_valid(planer, max_acc, dt):
    # Generate waypoints from track
    tracjetory = []
    for time in np.arange(0, planer.TS[-1], step=dt):
        state = planer.getStateAtTime(time)
        tracjetory.append(state)

    for p in tracjetory:
        acc_norm = np.linalg.norm(p.acc)
        if acc_norm > max_acc:
            return False
    return True

if __name__ == "__main__":
    T = convert_for_planer("assets/tracks/thesis-tracks/straight_track.csv")
    visualize_points(T)
    plt.show()
