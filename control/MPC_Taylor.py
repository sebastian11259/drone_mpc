"""
Standard MPC for Passing through a dynamic gate
"""
import casadi as ca
import numpy as np
import time
from os import system


#
# from high_mpc.common.quad_index import *

#
class MPC(object):
    """
    Nonlinear MPC - edited based on https://github.com/uzh-rpg/high_mpc/blob/master/high_mpc/mpc/mpc.py
    """

    def __init__(self, T, dt, so_path='./nmpc.so'):
        """
        Nonlinear MPC for quadrotor control
        """
        self.so_path = so_path

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
            1000, 1000, 1000,
            0, 0, 0])
        # cost matrix for the action
        self._Q_u = np.diag([0.000001, 0.000001, 0.000001, 0.000001])  #

        # initial state and control action
        self._quad_s0 = [0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._quad_u0 = [14468.4, 14468.4, 14468.4, 14468.4]  # Hover RPM

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


        # -- point to linearize
        px_, py_, pz_ = ca.SX.sym('px_'), ca.SX.sym('py_'), ca.SX.sym('pz_')
        #
        qw_, qx_, qy_, qz_ = ca.SX.sym('qw_'), ca.SX.sym('qx_'), ca.SX.sym('qy_'), \
            ca.SX.sym('qz')
        #
        vx_, vy_, vz_ = ca.SX.sym('vx_'), ca.SX.sym('vy_'), ca.SX.sym('vz_')

        #
        wx_, wy_, wz_ = ca.SX.sym('wx_'), ca.SX.sym('wy_'), ca.SX.sym('wz_')

        # -- conctenated vector
        self._x_ = ca.vertcat(px_, py_, pz_, qw_, qx_, qy_, qz_, vx_, vy_, vz_, wx_, wy_, wz_)

        # -- inputs to linearize

        r1_, r2_, r3_, r4_ = ca.SX.sym('r1_'), ca.SX.sym('r2_'), \
            ca.SX.sym('r3_'), ca.SX.sym('r4_')

        T_ = self.KF * (r1_ ** 2 + r2_ ** 2 + r3_ ** 2 + r4_ ** 2) / self.m
        Mx_ = self.d * self.KF * (r2_ ** 2 - r4_ ** 2)
        My_ = self.d * self.KF * (-r1_ ** 2 + r3_ ** 2)
        Mz_ = self.KM * (-r1_ ** 2 + r2_ ** 2 - r3_ ** 2 + r4_ ** 2)


        self._u_ = ca.vertcat(r1_, r2_, r3_, r4_)


        # # # # # # # # # # # # # # # # # # #
        # --------- Control Command ------------
        # # # # # # # # # # # # # # # # # # #

        r1, r2, r3, r4 = ca.SX.sym('r1'), ca.SX.sym('r2'), \
            ca.SX.sym('r3'), ca.SX.sym('r4')

        Jx = 1.4e-5
        Jy = 1.4e-5
        Jz = 2.17e-5


        thrust = self.KF * (r1 ** 2 + r2 ** 2 + r3 ** 2 + r4 ** 2) / self.m
        Mx = self.d * self.KF * (r2 ** 2 - r4 ** 2)
        My = self.d * self.KF * (-r1 ** 2 + r3 ** 2)
        Mz = self.KM * (-r1 ** 2 + r2 ** 2 - r3 ** 2 + r4 ** 2)

        # -- conctenated vector
        self._u = ca.vertcat(r1, r2, r3, r4)

        # # # # # # # # # # # # # # # # # # #
        # --------- System Dynamics ---------
        # # # # # # # # # # # # # # # # # # #

        x_dot = ca.vertcat(
            vx_,
            vy_,
            vz_,
            0.5 * (-wx_ * qx_ - wy_ * qy_ - wz_ * qz_),
            0.5 * (wx_ * qw_ + wz_ * qy_ - wy_ * qz_),
            0.5 * (wy_ * qw_ - wz_ * qx_ + wx_ * qz_),
            0.5 * (wz_ * qw_ + wy_ * qx_ - wx_ * qy_),
            2 * (qw_ * qy_ + qx_ * qz_) * T_,
            2 * (qy_ * qz_ - qw_ * qx_) * T_,
            (qw_ * qw_ - qx_ * qx_ - qy_ * qy_ + qz_ * qz_) * T_ - self._gz,
            (Mx_ + Jz * wz_ * wy_ - Jy * wy_ * wz_) / Jx,
            (My_ + Jx * wx_ * wz_ - Jz * wz_ * wx_) / Jy,
            (Mz_ + Jy * wy_ * wx_ - Jx * wx_ * wy_) / Jz
        )

        # -- Taylor first order

        x_lin = ca.vertcat(
            vx - vx_,
            vy - vy_,
            vz - vz_,
            -0.5 * (wx_ * (qx - qx_) + wy_ * (qy - qy_) + wz_ * (qz - qz_) + qx_ * (wx - wx_) + qy_ * (wy - wy_) + qz_ * (wz - wz_)),
             0.5 * (wx_ * (qw - qw_) + wz_ * (qy - qy_) - wy_ * (qz - qz_) + qw_ * (wx - wx_) + qy_ * (wz - wz_) - qz_ * (wy - wy_)),
             0.5 * (wy_ * (qw - qw_) - wz_ * (qx - qx_) + wx_ * (qz - qz_) + qw_ * (wy - wy_) - qx_ * (wz - wz_) + qz_ * (wx - wx_)),
             0.5 * (wz_ * (qw - qw_) + wy_ * (qx - qx_) - wx_ * (qy - qy_) + qw_ * (wz - wz_) + qx_ * (wy - wy_) - qy_ * (wx - wx_)),
            2 * T_ * ( qy_ * (qw - qw_) + qz_ * (qx - qx_) + qw_ * (qy_ - qy_) + qx_ * (qz - qz_)) + 2 * (qw_ * qy_ + qx_ * qz_) * (thrust - T_),
            2 * T_ * (-qx_ * (qw - qw_) - qw_ * (qx - qx_) + qz_ * (qy_ - qy_) + qy_ * (qz - qz_)) + 2 * (qy_ * qz_ + qw_ * qx_) * (thrust - T_),
            0.5 * T_ * (qw_ * (qw - qw_) - qx_ * (qx - qx_) - qy_ * (qy - qy_) + qz_ * (qz - qz_)) + (qw_ ** 2 - qx_ ** 2 - qy_ ** 2 + qz_ ** 2) * (thrust - T_),
            ((Jz - Jy) / Jx) * (wz_ * (wy - wy_) + wy_ * (wz - wz_)) + (1 / Jx) * (Mx - Mx_),
            ((Jx - Jz) / Jy) * (wz_ * (wx - wx_) + wx_ * (wz - wz_)) + (1 / Jy) * (My - My_),
            ((Jy - Jx) / Jz) * (wy_ * (wx - wx_) + wx_ * (wy - wy_)) + (1 / Jz) * (Mz - Mz_)
        )

        # -- Linearized state space

        self.f = ca.Function('f', [self._x, self._u, self._x_, self._u_], [x_dot + x_lin])

        # # Fold
        F = self.sys_dynamics(self._dt)
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


        u_min = [self._rpm_min, self._rpm_min, self._rpm_min, self._rpm_min]
        u_max = [self._rpm_max, self._rpm_max, self._rpm_max, self._rpm_max]
        x_bound = ca.inf
        x_min = [-x_bound for _ in range(self._s_dim)]
        x_max = [+x_bound for _ in range(self._s_dim)]
        #
        g_min = [0 for _ in range(self._s_dim)]
        g_max = [0 for _ in range(self._s_dim)]

        P = ca.SX.sym("P", self._s_dim + (self._s_dim) * self._N + 4)
        X = ca.SX.sym("X", self._s_dim, self._N + 1)
        U = ca.SX.sym("U", self._u_dim, self._N)

        # -- Control in previous iteration

        self.U_ = ca.SX.sym('U_',4,self._N)
        self.U_[:,0] = P[-4:]
        self.U_[:,1:] = U[:,:self._N-1]

        X_next = fMap(X[:, :self._N], U, self.U_)


        # "Lift" initial conditions
        self.nlp_w += [X[:, 0]]
        self.nlp_w0 += self._quad_s0
        self.lbw += x_min
        self.ubw += x_max

        # # starting point.
        self.nlp_g += [X[:, 0] - P[0:self._s_dim]]
        self.lbg += g_min
        self.ubg += g_max


        print(self.nlp_g)


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
            self.nlp_w0 += self._quad_s0
            self.lbw += x_min
            self.ubw += x_max

            # Add equality constraint
            self.nlp_g += [X_next[:, k] - X[:, k + 1]]
            self.lbg += g_min
            self.ubg += g_max

        print("Nlp_w", len(self.nlp_w))
        print("Nlp_w0", len(self.nlp_w))
        print("Nlp_g", len(self.nlp_g))
        print(X_next.shape)
        print(X.shape)

        a = [1,2,3,4,5,6,7,8,9,0]




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
        # TODO - shouldn't it be x0 first elemnt from a tracjetory ???
        self.sol = self.solver(
            x0=self.nlp_w0,
            lbx=self.lbw,
            ubx=self.ubw,
            p=ref_states,
            lbg=self.lbg,
            ubg=self.ubg)
        #
        sol_x0 = self.sol['x'].full()
        opt_u = sol_x0[self._s_dim:self._s_dim + self._u_dim]

        # TODO - why like this, Warm initialization
        self.nlp_w0 = list(sol_x0[self._s_dim + self._u_dim:2 * (self._s_dim + self._u_dim)]) + list(
            sol_x0[self._s_dim + self._u_dim:])


        #
        x0_array = np.reshape(sol_x0[:-self._s_dim], newshape=(-1, self._s_dim + self._u_dim))

        # return optimal action, and a sequence of predicted optimal trajectory.
        return opt_u, x0_array

    def sys_dynamics(self, dt):
        M = 10  # refinement
        DT = dt / M
        X0 = ca.SX.sym("X", self._s_dim)
        U = ca.SX.sym("U", self._u_dim)
        X_ = ca.SX.sym("X_", self._s_dim)
        U_ = ca.SX.sym("U_", self._u_dim)
        # #
        X = X0
        # for _ in range(M):
        #     # --------- RK4------------
        #     k1 = DT * self.f(X, U, X0, U_)
        #     k2 = DT * self.f(X + 0.5 * k1, U, X0, U_)
        #     k3 = DT * self.f(X + 0.5 * k2, U, X0, U_)
        #     k4 = DT * self.f(X + k3, U, X0, U_)
        #     #
        #     X = X + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        #     # Fold

        # ------ Euler method
        for _ in range(M):
            X = X + DT * self.f(X, U, X0, U_)

        F = ca.Function('F', [X0, U, U_], [X])
        return F


if __name__ == "__main__":
    mpc = MPC(1, 0.1, so_path="race_rl/control/mpc.so")