import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from iminuit import Minuit

################################################################################
# DEFINING CHISQ LIKELIHOOD ####################################################
################################################################################
def chi2(x, mu, err):
    return np.sum((x - mu)**2 / err**2)

################################################################################
# DEFINING THE CONSTRAINTS ON PARAMETERS #######################################
################################################################################
def calculate_omega_BD(Omega_BD, F0, H0):
    return (Omega_BD + F0 / H0) * 6 / (F0 / H0)**2

def calculate_Omega_m(Omega_BD, Omega_r, Omega_k):
    return 1 - Omega_BD - Omega_k - Omega_r

################################################################################
# DEFINING THE EQUATION SYSTEM #################################################
################################################################################
def calculate_H(H0, F, F0, Psi, Psi0, omega_BD, Omega_m, Omega_r, Omega_k, z):
    sign_H = 1.0 if 2 * H0 + F0 > 0.0 else -1.0
    return 0.5 * (-F + sign_H * np.sqrt(F**2 * (1 + omega_BD * 2 / 3) + 4 * H0**2 * (Psi0 / Psi * (Omega_m * (1 + z)**3 + Omega_r * (1 + z)**4) + Omega_k * (1 + z)**2)))

def equation_system(z, y, H0, F0, Psi0, omega_BD, Omega_m, Omega_r, Omega_k):
    F, Psi, t = y
    H = calculate_H(H0, F, F0, Psi, Psi0, omega_BD, Omega_m, Omega_r, Omega_k, z)
    dF_dz = F**2/((1 + z) * H) + 3 * F / (1 + z) - 3 / (3 + 2 * omega_BD) * Psi0 / Psi * H0**2 / H * Omega_m * (1 + z)**2
    dPsi_dz = - F * Psi / (H * (1 + z))
    dt_dz = -1 / ((1 + z) * H)
    return [dF_dz, dPsi_dz, dt_dz]

def integrate(t0, H0, F0, Psi0, omega_BD, Omega_m, Omega_r, Omega_k, z_array, z_span):
    sol = solve_ivp(
        fun = equation_system,
        t_span = z_span,
        y0 = [F0, Psi0, t0],
        method = "RK45",
        t_eval = z_array,
        args = (H0, F0, Psi0, omega_BD, Omega_m, Omega_r, Omega_k)
    )
    return sol

################################################################################
# LOADING THE DATA #############################################################
################################################################################
dataH = np.loadtxt('H_All.txt')
obs_values = dataH[:, 1]
obs_values_err = dataH[:, 2]
z_line = dataH[:, 0]

################################################################################
# DEFINING INITIAL PARAMETER VALUES, RESPECTING ALL EQUALITIES #################
################################################################################
Omega_r = 1e-4
Omega_k = 0.001
H0 = 67.4
Psi0 = 1.0
F0 = 1e-4
Omega_BD = 0.7
omega_BD = calculate_omega_BD(Omega_BD, F0, H0)
Omega_m = calculate_Omega_m(Omega_BD, Omega_r, Omega_k)

################################################################################
# DEFINING THE COST FUNCTION ###################################################
################################################################################
def cost_function(Omega_k, H0, Psi0, F0, Omega_BD):
    omega_BD = calculate_omega_BD(Omega_BD, F0, H0)
    Omega_m = calculate_Omega_m(Omega_BD, Omega_r, Omega_k)
    if Omega_m < 0.0 or Omega_m > 1.0:
        return 1e10
    solution = integrate(0.0, H0, F0, Psi0, omega_BD, Omega_m, Omega_r, Omega_k, z_line, [0.0, 2.5])
    if not solution.success or np.any(np.isnan(solution.y[0])):
        return 1e10
    H_line = calculate_H(H0, solution.y[0], F0, solution.y[1], Psi0, omega_BD, Omega_m, Omega_r, Omega_k, z_line)
    if np.any(np.isnan(H_line)):
        return 1e10
    return chi2(H_line, obs_values, obs_values_err)

m = Minuit(cost_function, Omega_k = Omega_k, H0 = H0, Psi0 = Psi0, F0 = F0, Omega_BD = Omega_BD)
m.limits = [(-1.0, 1.0), (40.0, 80.0), (0.01, 10.0), (0.0, 10.0), (0.0, 1.0)]

################################################################################
# DEFINING FIXED PARAMETERS ####################################################
################################################################################
# The lines below are very useful functionality of the MINUIT.
# MINUIT allows user to fix some of the parameters in a very easy fashion.
# In order to fix particular parameter to its initial value, just uncomment the relevant line.
# By default, all of the parameters are varied.

# m.fixed["Omega_k"] = True
# m.fixed["H0"] = True
# m.fixed["Psi0"] = True
# m.fixed["F0"] = True
# m.fixed["Omega_BD"] = True


################################################################################
# RUNNING THE OPTIMIZATION #####################################################
################################################################################
print("\n\nINITIAL PARAMETERS:")
print(m.params)
print("\n\nRUNNING MINUIT GRADIENT DESCENT")
print(m.migrad())

Omega_k = m.values["Omega_k"]
H0 = m.values["H0"]
Psi0 = m.values["Psi0"]
F0 = m.values["F0"]
Omega_BD = m.values["Omega_BD"]
omega_BD = calculate_omega_BD(Omega_BD, F0, H0)
Omega_m = calculate_Omega_m(Omega_BD, Omega_r, Omega_k)

print("\n\nFINAL STATS:")
print(
    f"Omega_m:  {Omega_m}\n"
    f"Omega_k:  {Omega_k}\n"
    f"Omega_r:  {Omega_r}\n"
    f"Omega_BD: {Omega_BD}\n"
    f"Omega_H0: {H0}\n"
    f"Omega_F0: {F0}\n"
    f"Psi0:     {Psi0}\n"
    f"omega_BD: {omega_BD}\n"
)

################################################################################
# PLOTTING THE RESULTS #########################################################
################################################################################
s_to_Gyr = 1 / 3.15e16
km_to_Mpc = 1e3 / 3.086e22

z_test_line = np.linspace(0.0, 2.5, 1000)
final_solution = integrate(0.0, H0, F0, Psi0, omega_BD, Omega_m, Omega_r, Omega_k, z_test_line, [0.0, 2.5])
F_final = final_solution.y[0]
Psi_final = final_solution.y[1]
t_final = final_solution.y[2] * s_to_Gyr / km_to_Mpc
z_final = final_solution.t
a_final = 1 / (1 + z_final)
H_final = calculate_H(H0, F_final, F0, Psi_final, Psi0, omega_BD, Omega_m, Omega_r, Omega_k, z_test_line)

plt.figure("The System of coupled odes", figsize = (10, 15))
plt.subplot(311)
plt.plot(t_final, a_final, label="$a$", color="black")
plt.xlabel("t")
plt.ylabel("$a$")


plt.subplot(312)
plt.plot(t_final, Psi_final, label="$\psi$", color="black")
plt.xlabel("t")
plt.yscale("log")
plt.ylabel("$\psi$")

plt.subplot(313)
plt.plot(t_final, F_final, label="$F$", color="black")
plt.xlabel("t")
plt.ylabel("$F$")
plt.yscale("log")
plt.tight_layout()
plt.savefig("pic_a_Psi_F.png")

plt.figure("H function")
plt.plot(z_final, H_final, label="$H$", color="blue")
plt.scatter(z_line, obs_values, label="$H_{exp}$", color="red")
plt.xlabel("z")
plt.ylabel("$H$")
plt.savefig("pic_H.png", bbox_inches="tight", dpi=500)
