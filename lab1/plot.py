import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from pathlib import Path
from os.path import join

# Vars
ABS_PATH = Path(__file__).parent
IN_PATH = join(ABS_PATH, "in")
OUT_PATH = join(ABS_PATH, "out")


#
# Model definitions
#

def linear_model(U, a1, a0):
    # 2. Linear function: I(U) = a0 + a1 * U
    return a1 * U + a0


def quadratic_model(U, a2, a1, a0):
    return a2 * U**2 + a1 * U + a0


def process_electrolyzer():
    data_in = join(IN_PATH, "lab1_elektrolyzer.data")
    U, I, = load_data(data_in)

    perr, popt = model_fit(linear_model, U, I)
    a1, a0 = popt

    print_report(popt, perr)
    electrolyzer_breakdown_voltage(a1, a0)

    plot_graph(U, I, popt, a0, a1, 'Voltampérová char. paliv. elektrolyzéru: $I(U)$')

    out_name = "lab1_clanek_output.pdf"
    data_out = join(OUT_PATH, out_name)
    save_graph(data_out, out_name)


def process_fuel_cell():
    data_in = join(IN_PATH, "lab1_clanek.data")
    U, I = load_data(data_in)

    perr, popt = model_fit(linear_model, U, I)
    a1, a0 = popt

    print_report(popt, perr)

    plot_graph(U, I, popt, a0, a1, 'Voltampérová char. paliv. článku: $I(U)$')

    maximum_fuel_cell_wattage(U, I)

    out_name = "lab1_elektrolyzer_output.pdf"
    data_out = join(OUT_PATH, out_name)
    save_graph(data_out, out_name)


def electrolyzer_breakdown_voltage(a1, a0):
    U_r = - (a0 / a1)

    print(f"Electrolyzer breakdown voltage: {U_r:.2f}V")
    print("-" * 30)


def maximum_fuel_cell_wattage(U, I):
    P = U * I

    perr, popt = model_fit(quadratic_model, U, P)
    a2, a1, a0 = popt
    print_report(popt, perr)

    plot_graph_wattage(U, P, popt, a0, a1, a2)

    out_name = "lab1_clanek_output_wattage.pdf"
    data_out = join(OUT_PATH, out_name)
    save_graph(data_out, out_name)


def load_data(load_str):
    data = np.loadtxt(load_str)
    U = data[:, 0]  # First column: Voltage
    I = data[:, 1]  # Second column: Current

    return U, I


# 3. Curve model fit
def model_fit(model_type, data_x, data_y):
    # popt = optimal parameters, pcov = covariance matrix (for errors)
    popt, pcov = curve_fit(model_type, data_x, data_y)
    perr = np.sqrt(np.diag(pcov)) # Standard deviations for errors

    return perr, popt


# 4. Report print
def print_report(popt, err):
    print("-" * 30)
    print("FIT RESULTS:")
    for i, val in enumerate(popt):
        print(f"Parametr a{len(popt)-1-i}: {val:.4f} ± {err[i]:.4f}")
    print("-" * 30)


# 5. Matplotlib plotting
def plot_graph(data_x, data_y, popt, a0, a1, label):
    plt.figure(figsize=(8, 5))

    # Plot the raw data points
    plt.scatter(data_x, data_y, color='black', label='Změřené data', marker='o', s=20)

    # Plot the fit line
    U_range = np.linspace(min(data_x), max(data_x), 100)
    plt.plot(U_range, linear_model(U_range, *popt), 'r-',
             label=f'Metoda nejmenších čtverců: $I = {a1:.2f}U + {a0:.2f}$')

    plt.xlabel('Napětí $U$ [V]')
    plt.ylabel('Proud $I$ [A]')
    plt.title(label)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()


def plot_graph_wattage(data_x, data_y, popt, a0, a1, a2):
    plt.figure(figsize=(8, 5))

    plt.scatter(data_x, data_y, color='black', label='Změřené data (P = U * I)', marker='o', s=20)

    U_range = np.linspace(min(data_x), max(data_x), 100)
    plt.plot(U_range, quadratic_model(U_range, *popt), 'r-',
             label=f'Metoda nejmen. čtverců (Parabola) $P = {a2:.2f}U^2 + {a1:.2f}U + {a0:.2f}$')

    plt.xlabel("Napětí $U$ [V]")
    plt.ylabel('Výkon $P$ [W]')
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