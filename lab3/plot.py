import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from pathlib import Path
from os.path import join

# Constants
ABS_PATH = Path(__file__).parent
IN_PATH = join(ABS_PATH, "in")
OUT_PATH = join(ABS_PATH, "out")

PRECISION = 6
PLOT_PRECISION = 3

# Greisinger G1710 specs
TEMPERATURE_DELTA = 0.1
TEMPERATURE_DIGIT_PRECISION = 1
TEMPERATURE_PRECISION = 0.05

# GW Instek GDM-8342 specs
RESISTANCE_DELTA = 0.01
RESISTANCE_DIGIT_PRECISION = 4
RESISTANCE_PRECISION = 0.1


#
# Model definitions
#

def linear_model(t, a1, a0):
    # Linear function: R(t) = b + a * t
    return a1 * t + a0


#
# Processing functions
#

def process(filename, label):
    data_in = join(IN_PATH, filename)
    t, R = load_data(data_in)

    u_BR = calc_type_BR_uncertainty(R)
    u_Bt = calc_type_Bt_uncertainty(t)

    slope, _ = np.polyfit(t, R, 1)
    sigma_combined = calc_type_B_combined(u_BR, u_Bt, slope)

    perr, popt = model_fit(linear_model, t, R, sigma_combined)

    print_report(popt, perr, label)
    calc_temperature_coef(popt, perr, label)

    return popt, t, R


#
# Calculation functions
#

def calc_type_Bt_uncertainty(t):
    # Calculation of B-type uncertainty for Temperature
    delta_max = (TEMPERATURE_PRECISION / 100) * np.abs(t) + (TEMPERATURE_DIGIT_PRECISION * TEMPERATURE_DELTA)
    u_B = delta_max / np.sqrt(3)
    return u_B


def calc_type_BR_uncertainty(R):
    # Calculation for B-type uncertainty for Resistance
    delta_max = (RESISTANCE_PRECISION / 100) * np.abs(R) + (RESISTANCE_DIGIT_PRECISION * RESISTANCE_DELTA)
    u_B = delta_max / np.sqrt(3)
    return u_B


def calc_type_B_combined(u_BR, u_Bt, slope):
    # Calculation of combined B-type uncertainty from initial linear fit
    # and Gauss's law of uncertainty propagation
    u_B_comb = np.sqrt(u_BR**2 + (slope * u_Bt)**2)
    return u_B_comb


def calc_temperature_coef(popt, perr, label):
    a1, a0 = popt
    u_c_a1, u_c_a0 = perr

    # Temperature coefficient
    alpha = a1 / a0

    # Gauss's law of uncertainty
    u_alpha = alpha * np.sqrt((u_c_a1 / a1)**2 + (u_c_a0 / a0)**2)

    print(f"Teplotní součinitel {label}: alpha = ({alpha * 1e3:.4f} ± {u_alpha * 1e3:.4f}) * 10^-3 K^-1")
    return alpha, u_alpha


#
# Helper functions
#

def load_data(load_str):
    data = np.loadtxt(load_str)
    t = data[:, 0]  # First column: Temperature
    R = data[:, 1]  # Second column: Resistance

    return t, R


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


def print_report(popt, err, label):
    print("-" * 30)
    print(f"--- FIT RESULTS ({label}) ---")
    print(f"Sklon (a1): {popt[0]:.{PRECISION}f} ± {err[0]:.{PRECISION}f}")
    print(f"Průsečík (a0): {popt[1]:.{PRECISION}f} ± {err[1]:.{PRECISION}f}")
    print("-" * 30)


def plot_and_save(
        cu_data,
        pt_data,
        popt_cu,
        popt_pt,
    ):
    out_name = "lab1_plot.pdf"
    data_out = join(OUT_PATH, out_name)

    def init_cust_label(popt, label):
        new_label = f'{label}: $R = {popt[0]:.{PLOT_PRECISION}f}t + {popt[1]:.{PLOT_PRECISION}f}$'
        new_label = new_label.replace(".", ",").replace("+ -", "-")
        return new_label


    t_cu, r_cu = cu_data
    t_pt, r_pt = pt_data

    plt.figure(figsize=(10, 7))

    # Cu data render
    t_range_cu = np.linspace(min(t_cu), max(t_cu), 100)
    label_cu = init_cust_label(popt_cu, "Cu")


    plt.scatter(t_cu, r_cu, color='red', label='Měď (data)', marker='o', alpha=0.7)
    plt.plot(t_range_cu, linear_model(t_range_cu, *popt_cu), 'r--',
             label=label_cu)

    # Pt data render
    t_range_pt = np.linspace(min(t_pt), max(t_pt), 100)
    label_pt = init_cust_label(popt_pt, "Pt")

    plt.scatter(t_pt, r_pt, color='blue', label='Platina (data)', marker='s', alpha=0.7)
    plt.plot(t_range_pt, linear_model(t_range_pt, *popt_pt), 'b-', label=label_pt)

    plt.title("Teplotní závislosti odporu $R$ na teplotě $t$ u Cu a Pt")
    plt.xlabel(r'Teplota $\frac{t}{°C}$')
    plt.ylabel(r'Odpor $\frac{R}{\Omega}$')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize='small')

    plt.tight_layout()
    save_graph(data_out, out_name)
    plt.show()


def save_graph(save_path, save_name):
    plt.savefig(save_path)
    print(f"Plot saved as '{save_name}'")


if __name__ == "__main__":
    print("Processing Copper data...")
    popt1, t1, r1 = process("lab3_copper.data", "Cu")

    print("Processing Platinum data...")
    popt2, t2, r2 = process("lab3_platinum.data", "Pt")

    plot_and_save((t1, r1), (t2, r2), popt1, popt2)
