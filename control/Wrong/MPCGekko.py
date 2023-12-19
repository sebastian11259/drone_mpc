import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt
import time


class MPC(object):

    def __init__(self, T, dt):
        self.m = GEKKO(remote=False)

        self.T = T
        self.dt = dt
        self.N = int(self.T / self.dt)
        self._rpm_min = 0.0
        self._rpm_max = 21702.0
        self._rpm_init = 14468.4

        # CONST

        self.CT = self.m.Const(3.1582e-10, 'CT')
        self.CD = self.m.Const(7.9379e-12, 'CD')
        self.d = self.m.Const(0.0397, 'd')
        self.ma = self.m.Const(0.027, 'ma')
        self.KF = self.m.Const(3.16e-10,'KF')
        self.KM = self.m.Const(7.94e-12, 'KM')
        self.gz = self.m.Const(9.81, 'gz')

        self.Jx = self.m.Const(1.4e-5, 'Jx')
        self.Jy = self.m.Const(1.4e-5, 'Jy')
        self.Jz = self.m.Const(2.17e-5, 'Jz')


        # X vect

        self.px, self.py, self.pz = self.m.CV('px'), self.m.CV('py'), self.m.CV('pz')
        self.qw, self.qx, self.qy, self.qz = self.m.SV('qw'), self.m.SV('qx'), self.m.SV('qy'),  self.m.SV('qz')
        self.vx, self.vy, self.vz = self.m.SV('vx'), self.m.SV('vy'), self.m.SV('vz')
        self.wx, self.wy, self.wz = self.m.SV('wx'), self.m.SV('wy'), self.m.SV('wz')

        # U vect

        self.u1, self.u2, self.u3, self.u4 = self.m.MV('u1'), self.m.MV('u2'), self.m.MV('u3'), self.m.MV('u4')

        for u in [self.u1, self.u2, self.u3, self.u4]:
            u.STATUS = 1
            u.FSTATUS = 0
            u.LOWER = self._rpm_min
            u.UPPER = self._rpm_max
            u.VALUE = self._rpm_init

        # X init

        x_init = [0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        status = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        fstatus = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        for i, x in enumerate([self.px, self.py, self.pz, self.qw, self.qx, self.qy, self.qz, self.vx, self.vy, self.vz, self.wx, self.wy, self.wz]):
            x.VALUE = x_init[i]
            # x.STATUS = status[i]
            # x.FSTATUS = fstatus[i]

        self.px.STATUS = 1
        self.py.STATUS = 1
        self.pz.STATUS = 1
        self.px.TR_INIT = 1
        self.py.TR_INIT = 1
        self.pz.TR_INIT = 1


        self.thrust = self.m.Intermediate(
            self.KF * (self.u1 ** 2 + self.u2 ** 2 + self.u3 ** 2 + self.u4 ** 2) / self.ma, 'thrust')
        self.Mx = self.m.Intermediate(self.d * self.KF * (self.u2 ** 2 - self.u4 ** 2), 'Mx')
        self.My = self.m.Intermediate(self.d * self.KF * (-self.u1 ** 2 + self.u3 ** 2), 'My')
        self.Mz = self.m.Intermediate(self.KM * (-self.u1 ** 2 + self.u2 ** 2 - self.u3 ** 2 + self.u4 ** 2), 'Mz')

        self.m.Equation(self.px.dt() == self.vx)
        self.m.Equation(self.py.dt() == self.vy)
        self.m.Equation(self.pz.dt() == self.vz)

        self.m.Equation(
            self.qw.dt() == 0.5 * (-self.wx * self.qx - self.wy * self.qy - self.wz * self.qz))
        self.m.Equation(
            self.qx.dt() == 0.5 * (self.wx * self.qw + self.wz * self.qy - self.wy * self.qz))
        self.m.Equation(
            self.qy.dt() == 0.5 * (self.wy * self.qw - self.wz * self.qx + self.wx * self.qz))
        self.m.Equation(
            self.qz.dt() == 0.5 * (self.wz * self.qw + self.wy * self.qx - self.wx * self.qy))

        self.m.Equation(self.vx.dt() == 2 * (self.qw * self.qy + self.qx * self.qz) * self.thrust)
        self.m.Equation(self.vy.dt() == 2 * (self.qy * self.qz - self.qw * self.qx) * self.thrust)
        self.m.Equation(self.vz.dt() == (
                    self.qw ** 2 - self.qx ** 2 - self.qy ** 2 - self.qz ** 2) * self.thrust - self.gz)

        self.m.Equation(self.wx.dt() == (
                    self.Mx + self.Jz * self.wy * self.wz - self.Jy * self.wy * self.wz) / self.Jx)
        self.m.Equation(self.wy.dt() == (
                    self.My + self.Jx * self.wx * self.wz - self.Jz * self.wx * self.wz) / self.Jy)
        self.m.Equation(self.wz.dt() == (
                    self.Mz + self.Jy * self.wx * self.wy - self.Jx * self.wx * self.wy) / self.Jz)




        self.m.options.IMODE = 6


        self.m.options.CV_TYPE = 2
        self.m.options.SOLVER = 1
        #
        # self.m.solver_options = [
        #     'tol 1e-5',
        #     'acceptable_tol 1e-5',
        #     'max_iter 500',
        #     # 'warm_start_init_point yes',
        #     'print_level 5',
        #     'linear_solver mumps'
        # ]

        self.m.time = np.linspace(0, self.T, self.N + 1)




    def solve( self, ref_states):

        t0 = time.time()






        self.px.VALUE = ref_states[0]
        self.py.VALUE = ref_states[1]
        self.pz.VALUE = ref_states[2]
        self.qw.VALUE = ref_states[3]
        self.qx.VALUE = ref_states[4]
        self.qy.VALUE = ref_states[5]
        self.qz.VALUE = ref_states[6]
        self.vx.VALUE = ref_states[7]
        self.vy.VALUE = ref_states[8]
        self.vz.VALUE = ref_states[9]
        self.wx.VALUE = ref_states[10]
        self.wy.VALUE = ref_states[11]
        self.wz.VALUE = ref_states[12]

        x = np.zeros(int(len(ref_states)/13)-1)
        y = np.zeros(int(len(ref_states)/13)-1)
        z = np.zeros(int(len(ref_states)/13)-1)
        for i in range(1, int(len(ref_states)/13)):
            x[i-1] = ref_states[i*13]
            y[i-1] = ref_states[i * 13 + 1]
            z[i-1] = ref_states[i * 13 + 1]



        self.px.SP = x
        self.py.SP = y
        self.pz.SP = z
        self.px.TAU = 5
        self.py.TAU= 5
        self.pz.TAU = 5
        # self.qw.TAU = 5

        # self.m.open_folder()

        self.m.solve(GUI=False, disp=True)

        print(time.time() - t0)

        # import json
        # with open(self.m.path + '//results.json') as f:
        #     results = json.load(f)
        #
        #
        print(self.u1.NEWVAL)
        print(self.u2.NEWVAL)
        print(self.u3.NEWVAL)
        print(self.u4.NEWVAL)
        print(self.u4.VALUE)
        #
        # plt.figure()
        # plt.subplot(2, 1, 1)
        # # plt.plot(m.time, p.value, 'b-', label='MV Optimized')
        # # plt.legend()
        # # plt.ylabel('Input')
        # plt.subplot(2, 1, 2)
        # plt.plot(self.m.time, results['v1.tr'], 'k-', label='Reference Trajectory')
        # # plt.plot(self.m.time, , 'r--', label='CV Response')
        # plt.ylabel('Output')
        # plt.xlabel('Time')
        # plt.legend(loc='best')
        # plt.show()
        #
        # print(len(ref_states))

        res = np.array([[self.u1.NEWVAL], [self.u2.NEWVAL], [self.u3.NEWVAL], [self.u4.NEWVAL]])
        print(ref_states[-13], ref_states[-12], ref_states[-11])
        print(self.px.VALUE)
        print(self.py.VALUE)
        print(self.pz.VALUE)

        return res




if __name__ == "__main__":
    mpc = MPC(1, 0.1)

    init = np.array([-5.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.999902007140843, -4.0098878095889764e-05, 1.9999896988008774, 0.999958129786947, 0.0, -0.009150640089505307,
     -6.677532329030582e-05, 0.004729552726675992, -0.0019159954626687186, -0.0004924868406092406, 0, 0, 0])



    a = mpc.solve(init)

# px, py, pz = m.Var('px'), m.Var('py'), m.Var('pz')
#
# qw, qx, qy, qz = m.Var('qw'), m.Var('qx'), m.Var('qy'),  m.Var('qz')
#
# vx, vy, vz = m.Var('vx'), m.Var('vy'), m.Var('vz')
#
# wx, wy, wz = m.Var('wx'), m.Var('wy'), m.Var('wz')