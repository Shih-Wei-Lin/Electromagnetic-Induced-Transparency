# Import required libraries
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
from numpy import linspace
import numpy as np
import scqubits as scq
from matplotlib import pyplot as plt
from labellines import labelLines
from hdf5Reader import hdf5Handle

def plot_transition_spectrum_data(fileDict, fluxonium, evals_count, point, zeroc, halfc, Phase=False):
    """
    Plot the transition spectrum data.

    Parameters:
    fileDict (str): Path to the HDF5 file.
    fluxonium (scq.Fluxonium): Fluxonium object from scqubits.
    evals_count (int): Number of eigenvalues to compute.
    point (int): Number of points for flux list.
    zeroc (float): Zero crossing current.
    halfc (float): Half quanta current.
    Phase (bool): Whether to plot phase or magnitude. Default is False.
    """
    
    def current(c):
        """Convert current to flux."""
        phi = c * (0.5 / (halfc - zeroc)) - zeroc * (0.5 / (halfc - zeroc))
        return phi

    # Import data from HDF5 file (use VNA)
    fig, axes = plt.subplots()
    data = hdf5Handle('yo', 'uA', fileDict)
    data.slice()
    
    if Phase:
        output_exp = np.angle(data.z)
    else:
        output_exp = np.abs(data.z)
        
    extent = current(min(data.y) / 10 ** -6), current(max(data.y) / 10 ** -6), data.x[0] / 10 ** 9, data.x[-1] / 10 ** 9
    plt.imshow(output_exp, origin="lower", aspect="auto", cmap="viridis", interpolation="None", extent=extent)
    
    # Solve eigen energy
    flux_list = linspace(current(min(data.y) / 10 ** -6), current(max(data.y) / 10 ** -6), point)
    eigval = fluxonium.get_spectrum_vs_paramvals("flux", flux_list, evals_count).energy_table
    eigval_list = eigval.T
    
    N = np.shape(eigval_list)[0]
    
    # Probable transitions: 0-f, 1-f
    spectrum_list_0 = np.zeros((N - 1, len(flux_list)))
    spectrum_list_1 = np.zeros((N - 2, len(flux_list)))
    
    for i in range(N - 1):
        spectrum_list_0[i] = eigval_list[i + 1] - eigval_list[0]
        
    for i in range(N - 2):
        spectrum_list_1[i] = eigval_list[i + 2] - eigval_list[1]
    
    for idx in range(np.shape(spectrum_list_0)[0]):
        plt.plot(flux_list, spectrum_list_0[idx], "--", alpha=0.5, label=f"{idx + 1},0")  # 0-f

    for idx in range(np.shape(spectrum_list_1)[0]):
        plt.plot(flux_list, spectrum_list_1[idx], "--", alpha=0.5, label=f"{idx + 2},1")  # 1-f
        
    plt.xlabel(r"$\frac{\phi_{ext}}{2\pi}$")
    plt.ylabel("Frequency (GHz)")
    plt.title("Transition spectrum", fontsize=16)
    plt.xlim([current(min(data.y) / 10 ** -6), current(max(data.y) / 10 ** -6)])
    plt.ylim([data.x[0] / 10 ** 9, data.x[-1] / 10 ** 9])
    labelLines(axes.get_lines(), zorder=1.5)
    
    fig, axes = fluxonium.plot_matelem_vs_paramvals('n_operator', 'flux', flux_list, select_elems=[(0, 1), (0, 2), (1, 2)])
    plt.xlim([current(min(data.y) / 10 ** -6), current(max(data.y) / 10 ** -6)])
    plt.ylabel("Matrix Element", fontsize=16)
    labelLines(axes.get_lines(), zorder=2.5)

# Fluxonium parameters
fluxonium = scq.Fluxonium(
    EJ=6.25,
    EC=0.75,
    EL=0.91,
    cutoff=30,
    flux=0.5
)

# File path and other parameters
fileDict = r'C:\Users\user\Downloads\2.hdf5'  # Input file path (only for VNA)
evals_count = 4

# Plot transition spectrum data
plot_transition_spectrum_data(fileDict, fluxonium, evals_count, 301, -491, -1220, Phase=False)  # (file path, fluxonium, evals_count, flux resolution, half quanta current (uA), integer quanta)
