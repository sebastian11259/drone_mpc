import numpy as np
from numpy import linalg as LA
from scipy import optimize

from collections import namedtuple
from tqdm import tqdm

DesiredState = namedtuple('DesiredState', 'pos vel acc jerk quat omega thrust torque')
        
class PathPlanner():
    def __init__(
        self, 
        waypoint: np.array, 
        max_velocity: float, 
        kt = 100,
        max_thrust: float = 0.59535, 
        max_torque: np.array =  np.array([0.008356, 0.008356, 0.007479]), 
    ):
        self.waypoint = waypoint
        self.max_velocity = max_velocity
        self.max_thrust = max_thrust * 0.9
        self.max_torque = max_torque * 0.9
        self.kt = kt
        # Order of the polynomial between 2 points
        self.order = 10
        n, dim = waypoint.shape
        self.len = n
        self.dim = dim
        self.TS = np.zeros(self.len)
        self.heading = waypoint[1] - waypoint[0]
        self.heading[2] = 0
        self.quat = quaterion_2_vectors(self.heading, [1, 0, 0])
        self.prev_omega= np.array([0, 0, 0])
        # Optimize track
        self.optimize()


    def callback(self, scale):
        time_scale = intermediate_result['x']
        scaled_T = self.T * time_scale
        self.TS[1:] = np.cumsum(scaled_T)
        self.cost, self.coef = self.minSnapTrajectory(scaled_T)

        traj = self.getTrajectory(0.01)

        if not self.isTracjetoryValid(traj):
            raise StopIteration
        else:
            self.scale = time_scale

    def optimize(self):
        relative = self.waypoint[0:-1] - self.waypoint[1:]
        T_init = LA.norm(relative, axis=-1) / self.max_velocity

        # Fist optimization, achive optimal time segments allocation
        T = optimize.minimize(
            self.getCost, 
            T_init, 
            method="COBYLA", 
            constraints= ({'type': 'ineq', 'fun': lambda T: T-T_init}))['x']
        # Second optimization chive optimal end time allocation, binary search
        max_time =  LA.norm(relative, axis=-1) / 0.2
        self.T = T
        T_scale = np.max(np.cumsum(max_time) / np.cumsum(T))
        lower_bound, upper_bound = 0, T_scale
        epsilon = 0.001
        # # Ensure the function values have different signs at the bounds
        max_iterations = 100
        for iteration in tqdm(range(max_iterations)): 
            mid = (lower_bound + upper_bound) / 2.0
            scaled_T = self.T * mid
            self.TS[1:] = np.cumsum(scaled_T)
            self.cost, self.coef = self.minSnapTrajectory(scaled_T)

            traj = self.getTrajectory(0.01)
            if not self.isTracjetoryValid(traj):
                lower_bound = mid
            else:
                upper_bound = mid

            if abs(upper_bound - lower_bound) < epsilon:
                break

        scale =  (lower_bound + upper_bound) / 2.00
        # scale = optimize.minimize(
        #     self.getTime, 
        #     T_scale,
        #     method='trust-constr',
        #     constraints=optimize.LinearConstraint(np.ones(1), lb=0),
        #     callback=self.callback
        # )['x']
        scale = 1
        self.scale = scale
        self.T *= scale
        self.TS[1:] = np.cumsum(self.T)
        self.cost, self.coef = self.minSnapTrajectory(self.T)


    def getTime(self, scale):
        return np.cumsum(self.T)[-2] * scale
        
    def getCost(self, T):
        cost, _ = self.minSnapTrajectory(T)
        cost += self.kt * np.sum(T)
        return cost

    def minSnapTrajectory(self, T):
        unkns = 4*(self.len - 2)

        Q = Hessian(T)
        A,B = self.getConstrains(T)

        invA = LA.inv(A)

        if unkns != 0:
            R = invA.T@Q@invA

            Rfp = R[:-unkns,-unkns:]
            Rpp = R[-unkns:,-unkns:]

            B[-unkns:,] = -LA.inv(Rpp)@Rfp.T@B[:-unkns,]

        P = invA@B
        cost = np.trace(P.T@Q@P)

        return cost, P

    def getConstrains(self, T):
        n = self.len-1
        order = self.order

        A = np.zeros((n*order, n*order))
        B = np.zeros((n*order, self.dim))

        B[:n,:] = self.waypoint[:-1, :]
        B[n:n*2,:] = self.waypoint[1:, :]

        # Way point constraints
        for i in range(n):
            A[i, order*i: order*(i+1)] = polyder(0)
            A[i+n, order*i: order*(i+1)] = polyder(T[i])

        #continuity contraints
        for i in range(n-1):
            A[2*n + 4*i: 2*n + 4*(i+1), order*i : order*(i+1)] = -polyder(T[i],'all')
            A[2*n + 4*i: 2*n + 4*(i+1), order*(i+1) : order*(i+2)] = polyder(0,'all')

        #start and end at rest
        A[6*n - 4 : 6*n, : order] = polyder(0,'all')
        A[6*n : 6*n + 4, -order : ] = polyder(T[-1],'all')

        #free variables
        for i in range(1,n):
            A[6*n + 4*i : 6*n + 4*(i+1), order*i : order*(i+1)] = polyder(0,'all')

        return A,B

    def getTrajectory(self, dt):
        """
        Default drone parementers are:  
        <mass value="0.027"/>
        <inertia ixx="1.4e-5" ixy="0.0" ixz="0.0" iyy="1.4e-5" iyz="0.0" izz="2.17e-5"/>
        """
        # Iterate the entire tracjetory and generate states allong the path        
        mass = 0.027
        J = np.array(
            [  
                [1.4e-5,     0,          0],
                [0,          1.4e-5,     0],
                [0,          0,          2.17e-5]
            ]
        )

        trajectory = []
        previous_quat = quaterion_2_vectors(self.heading, [1, 0, 0])
        previous_quat = np.array([previous_quat[3], previous_quat[0], previous_quat[1], previous_quat[2]])
        prev_omega = np.array([0, 0, 0])

        # TODO - should be from dt ???
        for t in np.arange(0, self.TS[-1], dt):
            if t >= self.TS[-1]: t = self.TS[-1] - 0.001
            i = np.where(t >= self.TS)[0][-1]

            t = t - self.TS[i]
            coeff = (self.coef.T)[:,self.order*i:self.order*(i+1)]

            pos  = coeff@polyder(t)
            vel  = coeff@polyder(t,1)
            accl = coeff@polyder(t,2)
            jerk = coeff@polyder(t,3)

            # Add gravityu to the acceleration
            g = [0, 0, 9.81]
            accl += g


            normalized_accl = accl / LA.norm(accl)
            # Calcula    
            # heading = vel / LA.norm(vel)
            heading = [1, 0, 0]
            projection = heading - np.dot(heading, normalized_accl) * normalized_accl
            # quat_xyz = np.crote a desired quaterion
            #heading = vel / LA.norm(vel)
            quat = quaterion_2_vectors(projection, [1, 0, 0],)
            quat = np.array([quat[3], quat[0], quat[1], quat[2]])
            quat /= LA.norm(quat)


            # Calculater desired omega
            # TODO - fix to be zero desired heading 
            omega = angular_velocities(previous_quat, quat, dt)
            previous_quat = quat

            # Calculate desired inputs
            d_omega = (omega - prev_omega) / dt
            torque = J @ d_omega
            prev_omega = omega
            thrus = accl * mass

            trajectory.append(DesiredState(pos, vel, accl, jerk, quat, omega, thrus, torque))

        return trajectory

    
    def getRefPath(self, t_start: float, dt: float, T: float):
        trajectory = []
        _N = int(T/dt)
        t_s = t_start + dt 
        # TODO - handling of and of episode ???
        for _ in range(_N):
            if t_s >= self.TS[-1]:
                trajectory.append([self.waypoint[-1],  [1, 0, 0, 0], [0, 0, 0], [0, 0, 0]])
                continue

            i = np.where(t_s >= self.TS)[0][-1]
            t = t_s - self.TS[i]
            coeff = (self.coef.T)[:,self.order*i:self.order*(i+1)]

            pos  = coeff@polyder(t)
            vel  = coeff@polyder(t,1)
            accl = coeff@polyder(t,2)
            jerk = coeff@polyder(t,3)

            # Add gravityu to the acceleration
            g = [0, 0, 9.81]
            accl += g

            normalized_accl = accl / LA.norm(accl)
            # Calcula    
            # heading = vel / LA.norm(vel)
            heading = [1, 0, 0]
            projection = heading - np.dot(heading, normalized_accl) * normalized_accl
            # quat_xyz = np.crote a desired quaterion
            #heading = vel / LA.norm(vel)
            quat = quaterion_2_vectors(projection, [1, 0, 0] )
            quat = np.array([quat[3], quat[0], quat[1], quat[2]])
            quat /= LA.norm(quat)

            # Calculater desired omega
            # Calculate desired inputs
            omega = [0, 0, 0]

            trajectory.extend([*pos, *quat, *vel, *omega])
            t_s += dt

        return trajectory
            


    def isTracjetoryValid(self, trajectory):
        for state in trajectory:
            thrust = state.thrust
            torque = state.torque

            if LA.norm(thrust) > self.max_thrust or np.any(np.abs(torque) > self.max_torque):
                return False

        return True

def angular_velocities(q1, q2, dt):
    return (2 / dt) * np.array([
        q1[3]*q2[0] - q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1], 
        q1[3]*q2[1] + q1[0]*q2[2] - q1[1]*q2[3] - q1[2]*q2[0], 
        q1[3]*q2[2] - q1[0]*q2[1] + q1[1]*q2[0] - q1[2]*q2[3]]
    )

def quaterion_2_vectors(u: np.array, v: np.array):
    k_cos_theta = np.dot(u, v);
    k = np.sqrt(LA.norm(u)**2 * LA.norm(v)**2)
    
    if k_cos_theta / k == -1:
        # 180 degree rotation around any orthogonal vector
        return np.array([*(u/LA.norm(u)), 0])

    quat = np.array([*np.cross(u, v), k_cos_theta + k])
    quat /= LA.norm(quat)
    return quat
 
def Hessian(T, order=10, opt=4):
    n = len(T)
    Q = np.zeros((n*order, n*order))
    for s in range(n):
        m = np.arange(0, opt, 1)
        for ii in range(order):
            for jj in range(order):
                if ii >= opt and jj >= opt:
                    pow = ii+jj-2*opt+1
                    # TODO: how to claculate a hessian
                    Q[order*s+ii,order*s+jj] = 2*np.prod((ii-m)*(jj-m))*T[s]**pow/pow

    return Q

def polyder(t, k = 0, order = 10):
    if k == 'all':
        terms = np.array([polyder(t,k,order) for k in range(1,5)])
    else:
        terms = np.zeros(order)
        coeffs = np.polyder([1]*order,k)[::-1]
        pows = t**np.arange(0,order-k,1)
        terms[k:] = coeffs*pows
    return terms



if __name__ == "__main__":
    from utils import convert_for_planer, visualize_points
    T = convert_for_planer("../assets/tracks/thesis-tracks/straight_track.csv")
    points = np.array(list(map(lambda x: x[0], T)))
    fig, ax = visualize_points(T)
    import matplotlib.pyplot as plt

    print("Calculating")
    pp = PathPlanner(points, max_velocity=10, kt=100)
    print("Finished")

    points = []
    for time in np.linspace(0.001, pp.TS[-1], 1000):
        state = pp.getStateAtTime(time)
        points.append((state.pos))
    
    x_coords, y_coords, z_coords = zip(*points)


    # plot the points as scatter
    ax.scatter(x_coords, y_coords, z_coords, c='g', marker='.')
    plt.show() 


            # self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (4*self.KF))
            # self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
            # if self.DRONE_MODEL == DroneModel.CF2X:
            #     self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)
    
            # self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)
            #             forces = np.array(rpm**2)*self.KF
            # torques = np.array(rpm**2)*self.KM
