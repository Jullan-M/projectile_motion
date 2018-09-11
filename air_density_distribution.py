__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
import numpy as np
from matplotlib import pyplot as plt

def rhofrac_uniform(y):
    return 1

def rhofrac_ideal_gas(y, y0 = 1.0e4):
    return np.exp(-y / y0)

def rhofrac_adiabatic(y, a=6.5e-3, alpha=2.5, T0=288.2):  # rho/rho0 as given in the appendix
    return np.power((1 - a / T0 * y), alpha)

y = np.arange(0, 45000, 1)

plt.figure(1)
plt.axhline(y=1, label=r"Uniform $\frac{\rho}{\rho_0}$", color="g", linestyle="-")
plt.plot(y/1000, rhofrac_ideal_gas(y), label=r"Ideal isothermal $\frac{\rho}{\rho_0}$", color="C1", linestyle="-")
plt.plot(y/1000, rhofrac_adiabatic(y), label=r"Adiabatic $\frac{\rho}{\rho_0}$", color="b", linestyle="-")
plt.title(r"Various air density distribution models")
plt.xlabel(r"$y$ (km)", fontsize=16)
plt.ylabel(r"$\frac{\rho}{\rho_0}$", fontsize=16)
plt.xlim(left = 0)
plt.ylim(bottom=0)
plt.legend()
plt.grid()
plt.savefig("air_density_distribution_models.pdf")
plt.show()