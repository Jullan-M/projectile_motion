__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
import numpy as np
from rk4 import Rk4
from matplotlib import pyplot as plt
from multiprocessing import Pool
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

class Projectile_2D:
    def __init__(self, x0, y0, v0, th0):  # Theta angle is in degrees.
        #   Initial values
        self.y0 = y0    # Height over origin (of reference point).
        self.x0 = x0    # Horizontal distance from origin.
        self.v0 = v0
        self.th0 = np.radians(th0)  # Converts degrees to radians.
        self.v0_vec = np.asarray( [self.v0*np.cos(self.th0), self.v0*np.sin(self.th0)] )
        self.t0 = 0

        #   Updating variables
        self.y = self.y0
        self.x = self.x0
        self.v = self.v0
        self.th = self.th0
        self.v_vec = np.asarray( [self.v0*np.cos(self.th0), self.v0*np.sin(self.th0)] )
        self.t = 0

    def __str__(self):
        return "y = " + str(self.y) + "\nx = " + str(self.x) + "\nv = " + str(self.v) + "\nvx = " + str(self.v_vec[0]) + "\nvy = " + str(self.v_vec[1]) + "\nth = " + str(np.rad2deg(self.th)) + "\nt = " + str(self.t)

    def reset(self):
        self.y = self.y0
        self.x = self.x0
        self.v = self.v0
        self.th = self.th0
        self.v_vec = np.asarray( [self.v0 * np.cos(self.th0), self.v0 * np.sin(self.th0)] )
        self.t = self.t0

    def update_th(self):
        self.th = np.arctan(self.v_vec[1]/self.v_vec[0])

    def update_v(self):
        self.v = np.sqrt(self.v_vec[0]**2+self.v_vec[1]**2)

    def set_th0(self, th_new):
        self.th0 = np.radians(th_new)
        self.reset()

def interpolate_xl( x0, x1, y0, y1):
    return (x0 - y0 * x1 / y1) / (- y0 / y1 + 1)

class Motion_2D:
    def __init__(self, projectile, dt):
        self.type = "N/A"
        self.projectile = projectile
        self.dt = dt

        #   Useful details about the motion.
        self.xl = 0
        self.y_max = 0
        self.tl = 0

        #   Arrays denoting the position of the projectile.
        self.x_arr = np.array([])
        self.y_arr = np.array([])

    def __str__(self):
        return self.type

    def __add__(self, other):
        return self.type + other

    #   Second order DEs for a projectile.
    #   Needed for Rk4-algorithm.
    def vx(self, t, x):
        return self.projectile.v_vec[0]

    def ax(self, t, vx):
        return 0

    def vy(self, t, y):
        return self.projectile.v_vec[1]

    def ay(self, t, vy):
        return 0

    def calculate_trajectory(self): #   Uses rk4-algorithm.
        #print("\nCalculating rk4 trajectory for a projectile with " + self.type + "...")
        x_arr0 = []
        y_arr0 = []
        #   Initializing objects of 4th order Runge-Kutta.
        Rk4_vx = Rk4(self.projectile.t0, self.projectile.x0, self.dt, self.vx)
        Rk4_ax = Rk4(self.projectile.t0, self.projectile.v_vec[0], self.dt, self.ax)
        Rk4_vy = Rk4(self.projectile.t0, self.projectile.y0, self.dt, self.vy)
        Rk4_ay = Rk4(self.projectile.t0, self.projectile.v_vec[1], self.dt, self.ay)

        while(self.projectile.y >= 0):  #    While the projectile is above ground.
            x_arr0.append(self.projectile.x)
            y_arr0.append(self.projectile.y)

            Rk4_vx.rk4()
            Rk4_ax.rk4()
            Rk4_vy.rk4()
            Rk4_ay.rk4()

            if (self.projectile.y < Rk4_vy.yi):
                self.y_max = Rk4_vy.yi

            self.projectile.t += self.dt
            self.projectile.x = Rk4_vx.yi
            self.projectile.v_vec[0] = Rk4_ax.yi
            self.projectile.y = Rk4_vy.yi
            self.projectile.v_vec[1] = Rk4_ay.yi
            self.projectile.update_th()
            self.projectile.update_v()

        #   Interpolation of landing point.
        n = len(x_arr0)
        self.xl = interpolate_xl(x_arr0[n - 1], self.projectile.x, y_arr0[n - 1], self.projectile.y)
        self.tl = self.projectile.t - self.dt / 2
        x_arr0.append(self.xl)
        y_arr0.append(0)

        #print("Final state of the projectile:\n")
        #print(self.projectile)
        self.projectile.reset()
        self.x_arr = np.array(x_arr0)
        self.y_arr = np.array(y_arr0)

    def calculate_final_state(self):
        #   Calculates ONLY the final state of the projectile. Probably not efficient at all.
        #   Nothing is appended to the trajectory member lists of this class.
        #   Does NOT reset projectile member class.
        #   Needed in optimize_proj_th0().
        x_old = 0.0
        y_old = 0.0
        #   Initializing objects of 4th order Runge-Kutta.
        Rk4_vx = Rk4(self.projectile.t0, self.projectile.x0, self.dt, self.vx)
        Rk4_ax = Rk4(self.projectile.t0, self.projectile.v_vec[0], self.dt, self.ax)
        Rk4_vy = Rk4(self.projectile.t0, self.projectile.y0, self.dt, self.vy)
        Rk4_ay = Rk4(self.projectile.t0, self.projectile.v_vec[1], self.dt, self.ay)

        while (self.projectile.y >= 0):  # While the projectile is above ground.
            x_old = self.projectile.x
            y_old = self.projectile.y

            Rk4_vx.rk4()
            Rk4_ax.rk4()
            Rk4_vy.rk4()
            Rk4_ay.rk4()

            self.projectile.t += self.dt
            self.projectile.x = Rk4_vx.yi
            self.projectile.v_vec[0] = Rk4_ax.yi
            self.projectile.y = Rk4_vy.yi
            self.projectile.v_vec[1] = Rk4_ay.yi
            self.projectile.update_th()
            self.projectile.update_v()
            if (y_old < self.projectile.y):
                self.y_max = self.projectile.y

        #   Interpolation of landing point.
        #   Note! Variables v_vec, t, and th of the projectile object will NOT necessarily be accurate for the landing position.
        self.xl = interpolate_xl(x_old, self.projectile.x, y_old, self.projectile.y)
        self.tl = self.projectile.t - self.dt / 2
        self.projectile.x = self.xl
        self.projectile.y = 0

    def optimize_proj_th0(self, th_min, th_max, dth):
        print("Optimizing projectile th0 for a motion with " + self.type + ".")
        th0_max = 0
        x_max = 0

        th = th_min
        while (th < th_max):
            self.projectile.set_th0(th)
            self.calculate_final_state()
            print(str(self.projectile.x) + "\t" + str(th))
            if (self.projectile.x > x_max):
                th0_max = th
                x_max = self.projectile.x
            th += dth

        self.projectile.set_th0(th0_max)
        print("Optimized th0: " + str(np.rad2deg(self.projectile.th0)))

class Motion_2D_nodrag(Motion_2D):
    def __init__(self, projectile, dt, g=9.81):
        super().__init__(projectile, dt)
        self.g = g
        self.type = "no drag"

    #   Second order DEs for a projectile with no drag.
    #   Needed for Rk4-algorithm.
    def ay(self, t, vy):
        return -self.g

    def calculate_analytic_trajectory(self):
        #print("\nCalculating analytic trajectory for projectile with NO drag...")
        x_arr0 = []
        y_arr0 = []
        y_old = 0
        x_old = 0
        while(self.projectile.y >= 0): #    While the projectile is above ground.
            x_old = self.projectile.x
            y_old = self.projectile.y
            x_arr0.append(x_old)
            y_arr0.append(y_old)

            self.projectile.t += self.dt
            #   The usual formulas for position and velocity for an object without drag
            self.projectile.x = self.projectile.x0 + self.projectile.v0_vec[0] * self.projectile.t
            self.projectile.y = self.projectile.y0 + self.projectile.v0_vec[1] * self.projectile.t - 0.5*self.g * self.projectile.t**2
            self.projectile.v_vec[1] = self.projectile.v0_vec[1] - self.g*self.projectile.t
            self.projectile.update_th()

            if (self.projectile.y > y_old):
                self.y_max = self.projectile.y

        #   Interpolation of landing point.
        #   NB! Variables v_vec, t, and th of the projectile object will NOT necessarily be accurate.
        n = len(x_arr0)
        self.xl = interpolate_xl(x_arr0[n-1], self.projectile.x, y_arr0[n-1], self.projectile.y)
        self.tl = self.projectile.t - self.dt / 2
        x_arr0.append(self.xl)
        y_arr0.append(0)

        #print("Details of projectile when it hits the ground:")
        #print(self.projectile)
        self.projectile.reset() #   Reset the projectile for future usage.
        self.x_arr = np.array(x_arr0)
        self.y_arr = np.array(y_arr0)

class Motion_2D_drag(Motion_2D): #  Assumes uniform air density everywhere.
    def __init__(self, projectile, dt, g=9.81, bpm = 4.00e-5):  #   Constants as given in compulsory exercise 1 description/appendix.
        super().__init__(projectile, dt)
        self.g = g
        self.bpm = bpm
        self.type = "uniform drag"

    #   Second-order DEs for a projectile with drag.
    #   Needed for Rk4-algorithm
    def ax(self, t, vx):
        return -self.bpm*vx**2

    def ay(self, t, vy):
        return -self.g-self.bpm*vy**2

class Motion_2D_drag_ideal_gas(Motion_2D_drag): #   Air density correction model 1: Isothermic ideal gas
    def __init__(self, projectile, dt, g=9.81, bpm = 4.00e-5, y0 = 1.0e4):  #   Constants as given in compulsory exercise 1 description/appendix.
        super().__init__(projectile, dt, g, bpm)
        self.y0 = y0
        self.type = "ideal drag"

    def rhofrac_ideal_gas(self):
        return np.exp(-self.projectile.y / self.y0)

    def ax(self, t, vx):
        return -self.bpm * vx ** 2 * self.rhofrac_ideal_gas()

    def ay(self, t, vy):
        return -self.g - self.bpm * vy ** 2 * self.rhofrac_ideal_gas()

class Motion_2D_drag_adiabatic(Motion_2D_drag): #   Air density correction model 2: Adiabatic approximation.
    def __init__(self, projectile, dt, g=9.81, bpm = 4.00e-5, a=6.5e-3, alpha=2.5, T0=288.2):   #   Constants as given in compulsory exercise 1 description/appendix. We assume that T0 = 15 C.
        super().__init__(projectile, dt, g, bpm)
        self.type = "adiabatic drag"
        self.a = a
        self.alpha = alpha
        self.T0 = T0

    def rhofrac_adiabatic(self):    #   rho/rho0 as given in the appendix
        return (1 - self.a / self.T0 * self.projectile.y) ** self.alpha

    def ax(self, t, vx):
        return -self.bpm * self.rhofrac_adiabatic() * vx ** 2

    def ay(self, t, vy):
        return -self.g - self.bpm * self.rhofrac_adiabatic() * vy ** 2

if (__name__ == '__main__'):
    #   Projectile setup
    pr1 = Projectile_2D( 0.0, 0.0, 700, 45.000 )
    pr2 = Projectile_2D( 0.0, 0.0, 700, 39.484 ) # x0,y0,v0,th0
    pr3 = Projectile_2D( 0.0, 0.0, 700, 46.757 )
    pr4 = Projectile_2D( 0.0, 0.0, 700, 44.938 )
    bertha = Projectile_2D( 0.0, 0.0, 1640, 54.9653 )


    #   NO DRAG
    #   Analytic motion
    mo1 = Motion_2D_nodrag( pr1, 0.01)
    #   Rk4-algorithm motion
    mo2 = Motion_2D_nodrag( pr1, 0.01)

    #   WITH DRAG, UNIFORM
    mo3 = Motion_2D_drag( pr2, 0.01)
    #mo3.optimize_proj_th0( 39.45, 39.50, 0.001 )

    #   WITH DRAG, ISOTHERMAL IDEAL GAS
    mo5 = Motion_2D_drag_ideal_gas(pr3, 0.01)
    #mo5.optimize_proj_th0( 46.75, 46.76, 0.001)

    #   WITH DRAG, ADIABATIC APPROXIMATION
    mo7 = Motion_2D_drag_adiabatic(pr4, 0.01)
    #mo7.optimize_proj_th0( 44.93, 44.94, 0.001)
        #   BIG BERTHA (PARIS GUN)
    mo8 = Motion_2D_drag_adiabatic(bertha, 0.01)
    # mo8.optimize_proj_th0( 54.96, 54.97, 0.0001)

    mo1.calculate_analytic_trajectory()
    mo2.calculate_trajectory()
    mo3.calculate_trajectory()
    mo5.calculate_trajectory()
    mo7.calculate_trajectory()
    mo8.calculate_trajectory()
    # Plotting
    plt.figure(1)
    plt.plot(mo1.x_arr / 1000, mo1.y_arr / 1000, label="Analytic, " + str(mo1) + r", $\theta =" + "%.2f" % np.rad2deg(mo1.projectile.th0) + r"^\circ$", color="b")
    plt.plot(mo2.x_arr / 1000, mo2.y_arr / 1000, label="Rk4, " + str(mo2) + r", $\theta =" + "%.2f" % np.rad2deg(mo2.projectile.th0) + r"^\circ$", color="r", linestyle="--")
    plt.title(r"Trajectory curves of projectile")
    plt.xlabel(r"$x$ (km)", fontsize=16)
    plt.ylabel(r"$y$ (km)", fontsize=16)
    plt.xlim(left = 0)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()
    plt.show()
    print()
    print("Description: no drag \t th0 = 45 deg \t v0 = 700 m/s")
    print("Analytic:\tLanding point: " + str(mo1.xl) + "\tTime of flight: " + str(mo1.tl) + "\tMaximum projectile height: " + str(mo1.y_max))
    print("Rk4:\t\tLanding point: " + str(mo2.xl) + "\tTime of flight: " + str(mo2.tl) + "\tMaximum projectile height: " + str(mo2.y_max))
    print()

    plt.figure(2)
    plt.plot(mo3.x_arr/1000, mo3.y_arr/1000, label=mo3 + r", $\theta =" + "%.3f" % np.rad2deg(mo3.projectile.th0) + r"^\circ$", color="r", linestyle="-")
    plt.plot(mo5.x_arr / 1000, mo5.y_arr / 1000, label=mo5 + r", $\theta =" + "%.3f" % np.rad2deg(mo5.projectile.th0) + r"^\circ$", color="g", linestyle="-")
    plt.plot(mo7.x_arr / 1000, mo7.y_arr / 1000, label=mo7 + r", $\theta =" + "%.3f" % np.rad2deg(mo7.projectile.th0) + r"^\circ$", color="m", linestyle="-")
    plt.title(r"Trajectory curves of projectiles with optimal $\theta_0$.")
    plt.xlabel(r"$x$ (km)", fontsize=16)
    plt.ylabel(r"$y$ (km)", fontsize=16)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()
    plt.show()
    print("Description: uniform quadratic drag \t optimal th0 = 39.484 deg \t v0 = 700 m/s")
    print("Landing point: " + str(mo3.xl) + "\tTime of flight: " + str(mo3.tl) + "\tMaximum projectile height: " + str(mo3.y_max))
    print()
    
    print("Description: isothermal ideal drag \t optimal th0 = 46.757 deg \t v0 = 700 m/s")
    print("Landing point: " + str(mo5.xl) + "\tTime of flight: " + str(mo5.tl) + "\tMaximum projectile height: " + str(mo5.y_max))
    print()

    print("Description: adiabatic drag \t optimal th0 = 44.938 deg \t v0 = 700 m/s")
    print("Landing point: " + str(mo7.xl) + "\tTime of flight: " + str(mo7.tl) + "\tMaximum projectile height: " + str(mo7.y_max))
    print()

    plt.figure(2)
    plt.plot(mo8.x_arr / 1000, mo8.y_arr / 1000, label=mo3 + r", $\theta =" + "%.3f" % np.rad2deg(mo8.projectile.th0) + r"^\circ$", color="r", linestyle="-")
    plt.title(r"Trajectory curve of Big Bertha cannon projectile")
    plt.xlabel(r"$x$ (km)", fontsize=16)
    plt.ylabel(r"$y$ (km)", fontsize=16)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()
    plt.show()
    print("BIG BERTHA LONG RANGE CANNON.")
    print("Description: adiabatic drag \t optimal th0 = 54.9653 deg \t v0 = 1640 m/s")
    print("Landing point: " + str(mo8.xl) + "\tTime of flight: " + str(mo8.tl) + "\tMaximum projectile height: " + str(mo8.y_max))
    print()


