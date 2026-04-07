import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from pathlib import Path
from os.path import join

# Vars
ABS_PATH = Path(__file__).parent
IN_PATH = join(ABS_PATH, "in")
OUT_PATH = join(ABS_PATH, "out")
DELTA = 0.001 # Delta for MASTECH MY-65
PRECISION = 6
PLOT_PRECISION = 3


#
# Calculation definitions
#

def calc_type_B_uncertainty():
    # Calculation of B-type uncertainty for Voltage and Amperage (same results, integrated into one function)
    u_B = DELTA / math.sqrt(12)
    return u_B


def calc_type_B_power(I, U, u_BI, u_BU):
    # Calculation of B-type uncertainty for Power
    u_P = np.sqrt((I**2 * u_BU**2) + (U**2 * u_BI**2))
    return u_P


#
# Model definitions
#

def linear_model(U, a1, a0):
    # 2. Linear function: I(U) = a0 + a1 * U
    return a1 * U + a0


def quadratic_model(U, a2, a1, a0):
    return a2 * U**2 + a1 * U + a0


#
# Processing functions
#

def process_electrolyzer():
    data_in = join(IN_PATH, "lab1_elektrolyzer.data")
    U, I, = load_data(data_in)

    u_BI = calc_type_B_uncertainty()
    sigma_I = np.full(len(I), u_BI)

    perr, popt = model_fit(linear_model, U, I, sigma_I)
    a1, a0 = popt
    u_c_a1, u_c_a0 = perr

    print_report(popt, perr)
    electrolyzer_breakdown_voltage(a1, a0, u_c_a1, u_c_a0)

    plot_graph(U, I, popt, a0, a1, 'Voltampérová char. elektrolyzéru: $I(U)$')

    out_name = "lab1_elektrolyzer_output.pdf"
    data_out = join(OUT_PATH, out_name)
    save_graph(data_out, out_name)


def process_fuel_cell():
    data_in = join(IN_PATH, "lab1_clanek.data")
    U, I = load_data(data_in)

    u_BI = calc_type_B_uncertainty()
    sigma_I = np.full(len(I), u_BI)

    perr, popt = model_fit(linear_model, U, I, sigma_I)
    a1, a0 = popt

    print_report(popt, perr)

    plot_graph(U, I, popt, a0, a1, 'Voltampérová char. paliv. článku: $I(U)$')

    out_name = "lab1_clanek_output.pdf"
    data_out = join(OUT_PATH, out_name)
    save_graph(data_out, out_name)

    maximum_fuel_cell_wattage_graph(U, I)


def maximum_fuel_cell_wattage_graph(U, I):
    P = U * I

    u_B_val = calc_type_B_uncertainty() # same for u_BI and u_BU
    sigma_P = calc_type_B_power(I, U, u_B_val, u_B_val)

    perr, popt = model_fit(quadratic_model, U, P, sigma_P)

    a2, a1, a0 = popt
    u_c_a2, u_c_a1, u_c_a0 = perr

    print_report(popt, perr)
    maximum_fuel_cell_wattage(a2, a1, a0, u_c_a2, u_c_a1, u_c_a0)
    plot_graph_wattage(U, P, popt, a0, a1, a2)

    out_name = "lab1_clanek_output_wattage.pdf"
    data_out = join(OUT_PATH, out_name)
    save_graph(data_out, out_name)


#
# Helper functions
#

def electrolyzer_breakdown_voltage(a1, a0, u_c_a1, u_c_a0):
    U_r = - (a0 / a1)
    u_U_r = math.sqrt((-(1 / a1) * u_c_a0)**2 + ((a0 / a1**2) * u_c_a1)**2)

    print(f"Electrolyzer breakdown voltage: {U_r:.{PRECISION}}V ± {u_U_r:.{PRECISION}f}")
    print("-" * 30)


def maximum_fuel_cell_wattage(a2, a1, a0, u_c_a2, u_c_a1, u_c_a0):
    P_max = a0 - (a1 ** 2) / (4 * a2)
    u_P_max = math.sqrt(u_c_a0**2 + (-(a1 / (2 * a2)) * u_c_a1)**2 + ((a1**2) / (4 * (a2**2)) * u_c_a2)**2)

    print(f"Maximum fuel cell wattage: {P_max:.{PRECISION}f}W ± {u_P_max:.{PRECISION}f}")
    print("-" * 30)


def load_data(load_str):
    data = np.loadtxt(load_str)
    U = data[:, 0]  # First column: Voltage
    I = data[:, 1]  # Second column: Current

    return U, I


# 3. Curve model fit
def model_fit(model_type, data_x, data_y, sigma_array):
    # popt = optimal parameters, pcov = covariance matrix (for errors)

    popt, pcov = curve_fit(
        model_type,
        data_x,
        data_y,
        sigma=sigma_array,
        absolute_sigma=True
    )
    perr = np.sqrt(np.diag(pcov)) # Standard deviations for errors

    return perr, popt


# 4. Report print
def print_report(popt, err):
    print("-" * 30)
    print("FIT RESULTS:")
    for i, val in enumerate(popt):
        print(f"Parametr a{len(popt)-1-i}: {val:.{PRECISION}f} ± {err[i]:.{PRECISION}f}")
    print("-" * 30)


# 5. Matplotlib plotting
def plot_graph(data_x, data_y, popt, a0, a1, label):
    plt.figure(figsize=(8, 5))

    # Plot the raw data points
    plt.scatter(data_x, data_y, color='black', label='Změřené data', marker='o', s=20)

    # Plot the fit line
    U_range = np.linspace(min(data_x), max(data_x), 100)
    cust_label = f'Metoda nejmenších čtverců: $I = {a1:.{PLOT_PRECISION}f}U + {a0:.{PLOT_PRECISION}f}$'
    cust_label = cust_label.replace(".", ",")
    plt.plot(U_range, linear_model(U_range, *popt), 'r-',
             label=cust_label)

    plt.xlabel(r'Napětí $\frac{U}{V}$')
    plt.ylabel(r'Proud $\frac{I}{A}$')
    plt.title(label)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()


def plot_graph_wattage(data_x, data_y, popt, a0, a1, a2):
    plt.figure(figsize=(8, 5))

    plt.scatter(data_x, data_y, color='black', label=r'Změřené data (P = $U \cdot I$)', marker='o', s=20)

    U_range = np.linspace(min(data_x), max(data_x), 100)

    cust_label = f'Metoda nejmen. čtverců (Parabola) $P = {a2:.{PLOT_PRECISION}f}U^2 + {a1:.{PLOT_PRECISION}f}U + {a0:.{PLOT_PRECISION}f}$'
    cust_label = cust_label.replace('.', ',')
    plt.plot(U_range, quadratic_model(U_range, *popt), 'r-',
             label=cust_label)

    plt.xlabel(r"Napětí $\frac{U}{V}$")
    plt.ylabel(r'Výkon $\frac{P}{W}$')
    plt.title('Výkon paliv. článku $P(U, I)$')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()


def save_graph(save_path, save_name):
    plt.savefig(save_path)
    print(f"Plot saved as '{save_name}'")


if __name__ == "__main__":
    print("Processing Electrolyzer data...")
    process_electrolyzer()

    print("Processing Clanek data...")
    process_fuel_cell()

    plt.show()
