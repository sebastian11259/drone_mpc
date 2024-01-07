"""
Standard MPC for Passing through a dynamic gate
"""
import casadi as ca
import numpy as np
import time
from os import system

from ray.rllib.algorithms.dt import DT


#
# from high_mpc.common.quad_index import *

#
class MPC(object):
    """
    Nonlinear MPC - edited based on https://github.com/uzh-rpg/high_mpc/blob/master/high_mpc/mpc/mpc.py
    """

    def __init__(self, T, dt, divisor = 4):
        """
        Nonlinear MPC for quadrotor control
        """
        self.divisor = divisor

        # Time constant
        self._T = T
        self._dt = dt
        self._N = int(self._T / self._dt)

        # Gravity
        self._gz = 9.81

        # Quadrotor constant
        # TODO - update constraints
        self._rpm_min = 0.0
        self._rpm_max = 21702.0

        # Motor paremets values
        self.CT = 3.1582e-10
        self.CD = 7.9379e-12
        self.d = 0.0397
        self.m = 0.027
        self.KF = 3.16e-10
        self.KM = 7.94e-12
        # self.KF = 3.1582e-10
        # self.KM = 7.9379e-12


        #
        # state dimension (px, py, pz,           # quadrotor position
        #                  qw, qx, qy, qz,       # quadrotor quaternion
        #                  vx, vy, vz,           # quadrotor linear velocity
        #                  wx, wy, wz            # quadrotor angular velocity
        self._s_dim = 13
        # action dimensions (rpm1, rmp2, rmp3, rmp4)
        self._u_dim = 4

        # cost matrix for tracking the goal point
        self._Q_track = np.diag([
            1000, 1000, 1000,  # delta_x, delta_y, delta_z
            0, 0, 0, 0,  # delta_qw, delta_qx, delta_qy, delta_qz
            100, 100, 100,
            0, 0, 0])

        self._Q_goal = np.diag([
            10000, 10000, 10000,  # delta_x, delta_y, delta_z
            0, 0, 0, 0,  # delta_qw, delta_qx, delta_qy, delta_qz
            100, 100, 100,
            0, 0, 0])
        # cost matrix for the action
        self._Q_u = np.diag([0.00001, 0.00001, 0.00001, 0.00001])  #

        # initial state and control action
        self._quad_s0 = [0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # self._quad_u0 = [14468.4, 14468.4, 14468.4, 14468.4]  # Hover RPM

        self._quad_u0 = [9.79, 0, 0, 0]  # Hover RPM


        # self.P1, self.P2, self.P3, self.P4, self.P5, self.P6 = 0, 0, 0, 0, 0, 0

        self._initDynamics()

    def _initDynamics(self, ):
        # # # # # # # # # # # # # # # # # # #
        # ---------- Input States -----------
        # # # # # # # # # # # # # # # # # # #

        px, py, pz = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz')
        #
        qw, qx, qy, qz = ca.SX.sym('qw'), ca.SX.sym('qx'), ca.SX.sym('qy'), \
            ca.SX.sym('qz')
        #
        vx, vy, vz = ca.SX.sym('vx'), ca.SX.sym('vy'), ca.SX.sym('vz')

        #
        wx, wy, wz = ca.SX.sym('wx'), ca.SX.sym('wy'), ca.SX.sym('wz')

        # -- conctenated vector
        self._x = ca.vertcat(px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz)

        # # # # # # # # # # # # # # # # # # #
        # --------- Control Command ------------
        # # # # # # # # # # # # # # # # # # #


        Jx = 1.4e-5
        Jy = 1.4e-5
        Jz = 2.17e-5

        thrust, Mx, My, Mz = ca.SX.sym('thrust'), ca.SX.sym('Mx'), ca.SX.sym('My'), ca.SX.sym('Mz')

        # r1, r2, r3, r4 = ca.SX.sym('r1'), ca.SX.sym('r2'), \
        #     ca.SX.sym('r3'), ca.SX.sym('r4')

        # thrust = self.KF * (r1 ** 2 + r2 ** 2 + r3 ** 2 + r4 ** 2) / self.m
        # Mx = self.d * self.KF * (r2 ** 2 - r4 ** 2)
        # My = self.d * self.KF * (-r1 ** 2 + r3 ** 2)
        # Mz = self.KM * (-r1 ** 2 + r2 ** 2 - r3 ** 2 + r4 ** 2)



        # Matrix to compute (r1-r4)**2
        inv = np.array([
            [self.m / (4 * self.KF), 0, -1 / (2 * self.d * self.KF), -1 / (4 * self.KM)],
            [self.m / (4 * self.KF), 1 / (2 * self.d * self.KF), 0, 1 / (4 * self.KM)],
            [self.m / (4 * self.KF), 0, 1 / (2 * self.d * self.KF), -1 / (4 * self.KM)],
            [self.m / (4 * self.KF), -1 / (2 * self.d * self.KF), 0, 1 / (4 * self.KM)]
        ])

        self.inv = ca.SX(inv)


        # Control Vector

        self._u = ca.vertcat(thrust, Mx, My, Mz)

        # Min-Max control

        u_min = [0,
                 self.d * self.KF * (self._rpm_min ** 2 - self._rpm_max ** 2),
                 self.d * self.KF * (self._rpm_min ** 2 - self._rpm_max ** 2),
                 self.KM * (-2 * self._rpm_max)
                 ]

        u_max = [self.KF * (4 * self._rpm_max**2) / self.m,
                 self.d * self.KF * (self._rpm_max ** 2 - self._rpm_min ** 2),
                 self.d * self.KF * (self._rpm_max ** 2 - self._rpm_min ** 2),
                 self.KM * (2 * self._rpm_max)
                 ]


        # Function to compute rpm from Thrust, Mx, My, mz

        self.rpm = ca.Function('rpm', [self._u], [np.sqrt(ca.fabs(self.inv @ self._u))] )


        # # # # # # # # # # # # # # # # # # #
        # --------- System Dynamics ---------
        # # # # # # # # # # # # # # # # # # #


        # Scheduling variables
        self.P1, self.P2, self.P3, self.P4, self.P5, self.P6 = ca.SX.sym('P1'), ca.SX.sym('P2'), ca.SX.sym('P3'), ca.SX.sym('P4'), ca.SX.sym('P5'), ca.SX.sym('P6')

        self.Pi = ca.vertcat(self.P1, self.P2, self.P3, self.P4, self.P5, self.P6)


        E = 0.01


        # A matrix

        self.A = ca.SX.zeros(13,13)

        self.A[0,7] = 1
        self.A[1,8] = 1
        self.A[2,9] = 1
        self.A[3,10], self.A[3,11], self.A[3,12] = -0.5*self.P2, -0.5*self.P3, -0.5*self.P4
        self.A[4,10], self.A[4,11], self.A[4,12] = 0.5*self.P1, -0.5*self.P4, 0.5*self.P3
        self.A[5,10], self.A[5,11], self.A[5,12] = 0.5*self.P4, 0.5*self.P1, -0.5*self.P2
        self.A[6,10], self.A[6,11], self.A[6,12] = -0.5*self.P3, 0.5*self.P2, 0.5*self.P1
        self.A[7, 5], self.A[7, 6] = E, E
        self.A[8, 4], self.A[8, 6] = E, E
        self.A[9, 4], self.A[9, 5] = E, E
        self.A[10,12] = (Jz-Jy)/Jx * self.P6
        self.A[11,12] = (Jx-Jz)/Jy * self.P5
        self.A[12,11] = (Jy-Jx)/Jz * self.P5

        # B matrix

        self.B = ca.SX.zeros(13,5)

        self.B[7,0] = 2*(self.P1 * self.P3 + self.P2*self.P4)
        self.B[8,0] = 2*(self.P3 * self.P4 - self.P1*self.P2)
        self.B[9,0], self.B[9,4] = (self.P1**2 - self.P2**2 - self.P3**2 + self.P4**2), -1
        self.B[10,1] = 1/Jx
        self.B[11,2] = 1/Jy
        self.B[12,3] = 1/Jz

        # self.fa = ca.Function('fa', [self.Pi], [self.A], ['Pi'], ['Ad'])
        # self.fb = ca.Function('fb', [self.Pi], [self.B], ['Pi'], ['Bd'])



        # Discrete A and B matrix with Pi parameter

        self.fa = ca.Function('fad', [self.Pi], [ca.SX.eye(13) +  (self.A * (self._dt/self.divisor))])
        self.fb = ca.Function('fbd', [self.Pi], [self.B * (self._dt/self.divisor)])

        # Weight matrix for error

        self.K = np.diag([0.2,0.2,0.2,0,0,0,0,0,0,0,0,0,0])


        # # Fold

        F = self.sys_dynamics2()
        fMap = F.map(self._N, "openmp")  # parallel

        # # # # # # # # # # # # # # #
        # ---- loss function --------
        # # # # # # # # # # # # # # #

        # placeholder for the quadratic cost function
        Delta_s = ca.SX.sym("Delta_s", self._s_dim)
        Delta_g = ca.SX.sym("Delta_g", self._s_dim)
        Delta_u = ca.SX.sym("Delta_u", self._u_dim)

        #
        cost_track = Delta_s.T @ self._Q_track @ Delta_s
        cost_u = Delta_u.T @ self._Q_u @ Delta_u
        cost_goal = Delta_g.T @ self._Q_goal @ Delta_g
        #
        f_cost_track = ca.Function('cost_track', [Delta_s], [cost_track])
        f_cost_goal = ca.Function('cost_goal', [Delta_g], [cost_goal])
        f_cost_u = ca.Function('cost_u', [Delta_u], [cost_u])

        #
        # # # # # # # # # # # # # # # # # # # #
        # # ---- Non-linear Optimization -----
        # # # # # # # # # # # # # # # # # # # #
        self.nlp_w = []  # nlp variables
        self.nlp_w0 = []  # initial guess of nlp variables
        self.lbw = []  # lower bound of the variables, lbw <= nlp_x
        self.ubw = []  # upper bound of the variables, nlp_x <= ubw
        #
        self.mpc_obj = 0  # objective
        self.nlp_g = []  # constraint functions
        self.lbg = []  # lower bound of constrait functions, lbg < g
        self.ubg = []  # upper bound of constrait functions, g < ubg

        x_bound = ca.inf
        x_min = [-x_bound for _ in range(self._s_dim)]
        x_max = [+x_bound for _ in range(self._s_dim)]
        # x_min = [-x_bound for _ in range(3)] + [-1, -1, -1, -1] + [-x_bound for _ in range(6)]
        # x_max = [x_bound for _ in range(3)] + [1, 1, 1, 1] + [x_bound for _ in range(6)]

        x_6_min = [-x_bound for _ in range(6)]
        x_6_max = [+x_bound for _ in range(6)]
        #
        g_min = [0 for _ in range(self._s_dim)]
        g_max = [0 for _ in range(self._s_dim)]
        g_6_min = [0 for _ in range(6)]
        g_6_max = [0 for _ in range(6)]

        # TODO -why +3
        P = ca.SX.sym("P", self._s_dim + (self._s_dim) * self._N + self._s_dim)
        X = ca.SX.sym("X", self._s_dim, self._N + 1)
        U = ca.SX.sym("U", self._u_dim, self._N)
        Pi = ca.SX.sym("Pi", 6, self._N + 1)
        #
        X_next = fMap(X[:, :self._N], U, Pi[:, :self._N], P[-13:])

        # "Lift" initial conditions
        self.nlp_w += [X[:, 0]]
        self.nlp_w += [Pi[:, 0]]
        self.nlp_w0 += self._quad_s0
        self.nlp_w0 += [1,0,0,0,0,0]
        self.lbw += x_min
        self.lbw += x_6_min
        self.ubw += x_max
        self.ubw += x_6_max


        # # starting point.
        self.nlp_g += [X[:, 0] - P[0:self._s_dim]]
        self.nlp_g += [Pi[0:4, 0] - P[3:7] ]
        self.nlp_g += [Pi[4:,0] - P[10:12] ]
        self.lbg += g_min
        self.lbg += g_6_min
        self.ubg += g_max
        self.ubg += g_6_max

        for k in range(self._N):
            #
            self.nlp_w += [U[:, k]]
            self.nlp_w0 += self._quad_u0
            self.lbw += u_min
            self.ubw += u_max

            # retrieve time constant
            # idx_k = self._s_dim+self._s_dim+(self._s_dim+3)*(k)
            # idx_k_end = self._s_dim+(self._s_dim+3)*(k+1)
            # time_k = P[ idx_k : idx_k_end]

            # cost for tracking the goal position
            cost_track, cost_goal_k, cost_gap_k = 0, 0, 0

            # TODO - Here can be an error
            # delta_s_k = (X[:, k] - P[self._s_dim+(self._s_dim)*k: self._s_dim+(self._s_dim)*(k+1):])
            if k < self._N:
                delta_s_k = (X[:, k + 1] - P[self._s_dim + (self._s_dim) * k:
                                             self._s_dim + (self._s_dim) * (k + 1):])
                cost_track = f_cost_track(delta_s_k)
            else:
                delta_s_k = (X[:, k + 1] - P[self._s_dim + (self._s_dim) * k:
                                             self._s_dim + (self._s_dim) * (k + 1):])
                cost_goal_k = f_cost_goal(delta_s_k)

            # TODO - can be wrong
            delta_u_k = U[:, k] - self._quad_u0
            cost_u_k = f_cost_u(delta_u_k)

            self.mpc_obj = self.mpc_obj + cost_goal_k + cost_u_k + cost_track

            # New NLP variable for state at end of interval
            self.nlp_w += [X[:, k + 1]]
            self.nlp_w += [Pi[:, k+1]]
            self.nlp_w0 += self._quad_s0
            self.nlp_w0 += [1, 0, 0, 0, 0, 0]
            self.lbw += x_min
            self.lbw += x_6_min
            self.ubw += x_max
            self.ubw += x_6_max

            # Add equality constraint
            self.nlp_g += [X_next[:, k] - X[:, k + 1]]
            self.nlp_g += [Pi[0:4, k+1] - X[3:7,k+1]]
            self.nlp_g += [Pi[4:, k+1] - X[10:12, k+1]]
            # self.nlp_g += [Pi[0:4, k+1] - P[3:7]]
            # self.nlp_g += [Pi[4:, k+1] - P[10:12]]
            self.lbg += g_min
            self.lbg += g_6_min
            self.ubg += g_max
            self.ubg += g_6_max



        # nlp objective
        nlp_dict = {'f': self.mpc_obj,
                    'x': ca.vertcat(*self.nlp_w),
                    'p': P,
                    'g': ca.vertcat(*self.nlp_g)}

        # # # # # # # # # # # # # # # # # # #
        # -- qpoases
        # # # # # # # # # # # # # # # # # # #
        # nlp_options ={
        #     'verbose': False, \
        #     "qpsol": "qpoases", \
        #     "hessian_approximation": "gauss-newton", \
        #     "max_iter": 100,
        #     "tol_du": 1e-2,
        #     "tol_pr": 1e-2,
        #     "qpsol_options": {"sparse":True, "hessian_type": "posdef", "numRefinementSteps":1}
        # }
        # self.solver = ca.nlpsol("solver", "sqpmethod", nlp_dict, nlp_options)
        # cname = self.solver.generate_dependencies("mpc_v1.c")
        # system('gcc -fPIC -shared ' + cname + ' -o ' + self.so_path)
        # self.solver = ca.nlpsol("solver", "sqpmethod", self.so_path, nlp_options)

        # # # # # # # # # # # # # # # # # # #
        # -- ipopt
        # # # # # # # # # # # # # # # # # # #
        ipopt_options = {
            'verbose': False, \
            "ipopt.tol": 1e-5,
            "ipopt.acceptable_tol": 1e-5,
            "ipopt.max_iter": 500,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": False
        }

        # # TODO - generate a c code file
        self.solver = ca.nlpsol("solver", "ipopt", nlp_dict, ipopt_options)

        # eee = self.sys_dynamics1()


        # self.solver = ca.qpsol("solver",'qpoases', nlp_dict)

        # # jit (just-in-time compilation)
        # print("Generating shared library........")
        # cname = self.solver.generate_dependencies("mpc_v1.c")
        # system('gcc -fPIC -shared -O3 ' + cname + ' -o ' + self.so_path) # -O3

        # # reload compiled mpc
        # print(self.so_path)
        # self.solver = ca.nlpsol("solver", "ipopt", self.so_path, ipopt_options)

    def solve(self, ref_states):
        # # # # # # # # # # # # # # # #
        # -------- solve NLP ---------
        # # # # # # # # # # # # # # # #

        self.sol = self.solver(
            x0=self.nlp_w0,
            lbx=self.lbw,
            ubx=self.ubw,
            p=ref_states,
            lbg=self.lbg,
            ubg=self.ubg)
        #
        sol_x0 = self.sol['x'].full()
        opt_u = sol_x0[self._s_dim + 6:self._s_dim + self._u_dim + 6]



        # TODO - why like this, Warm initialization
        self.nlp_w0 = list(sol_x0[self._s_dim + self._u_dim + 6 :2 * (self._s_dim + self._u_dim+ 6) ]) + list(
            sol_x0[self._s_dim + self._u_dim + 6:])


        x0_array = np.reshape(sol_x0[:-(self._s_dim+6)], newshape=(-1, self._s_dim + self._u_dim + 6))


        # return optimal action, and a sequence of predicted optimal trajectory.
        return self.rpm(opt_u), x0_array

    def sys_dynamics(self, dt):
        M = 10  # refinement
        DT = dt / M
        X0 = ca.SX.sym("X", self._s_dim)
        U = ca.SX.sym("U", self._u_dim)
        P = ca.SX.sym("P", 6)
        # #
        X = X0
        for _ in range(M):
            # --------- RK4------------
            k1 = DT * self.f(X, U, P)
            k2 = DT * self.f(X + 0.5 * k1, U, P)
            k3 = DT * self.f(X + 0.5 * k2, U, P)
            k4 = DT * self.f(X + k3, U, P)
            #
            X = X + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            # Fold
        F = ca.Function('F', [X0, U, P], [X])
        return F



    # def sys_dynamics1(self):
    #     X0 = ca.SX.sym("X", self._s_dim)
    #     X = ca.SX.sym("X", self._s_dim, self._N)
    #     P = ca.SX.sym("P", 6, self._N)
    #     U = ca.SX.sym("U", self._u_dim, self._N)
    #
    #     X[:, 0] = self.fa(P[:, 0]) @ X0 + self.fb(P[:,0]) @ ca.vertcat(U[:, 0], self._gz)
    #
    #     for i in range(1, self._N):
    #         X[:,i] = self.fa(P[:,i]) @ X[:, i-1] + self.fb(P[:,i]) @ ca.vertcat(U[:,i-1], self._gz)
    #
    #
    #
    #
    #     F = ca.Function('F', [X0, U, P], [X])
    #     return F

    # Prediction: X(k+1) = A(k)X(k) + B(k)U(k)
    def sys_dynamics2(self):
        X0 = ca.SX.sym("X", self._s_dim)
        U = ca.SX.sym("U", self._u_dim)
        P = ca.SX.sym("P", 6)
        E = ca.SX.sym("Err", self._s_dim)

        X=X0
        # #
        for i in range(self.divisor):
            X = self.fa(P) @ X + self.fb(P) @ ca.vertcat(U, self._gz)

        # X = X-(self.K@E)

        F = ca.Function('F', [X0, U, P, E], [X])
        return F


if __name__ == "__main__":
    mpc = MPC(1, 0.1)