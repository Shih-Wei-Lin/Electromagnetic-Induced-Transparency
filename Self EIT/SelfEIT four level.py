# Required imports
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# Define the Hamiltonian and collapse operators for the system
def H_sEIT(del_c, del_p, Omega_c, Omega_p, chi, kappa, Gamma):
    """
    Define the Hamiltonian and collapse operators for the EIT system.

    Parameters:
    del_c -- Detuning of the control field
    del_p -- Detuning of the probe field
    Omega_c -- Rabi frequency of the control field
    Omega_p -- Rabi frequency of the probe field
    chi -- Dispersive shift
    kappa -- Cavity decay rate
    Gamma -- Relaxation rate

    Returns:
    H -- Hamiltonian of the system
    collapse_ops -- List of collapse operators
    """
    # Basis states for the system
    term1 = (-del_c)*qt.basis(4, 1)*qt.basis(4, 1).dag() + \
            (chi-del_p)*qt.basis(4, 2)*qt.basis(4, 2).dag() + \
            (-chi-(del_c+del_p))*qt.basis(4, 3)*qt.basis(4, 3).dag()  # Energy terms

    term2 = (Omega_c/2)*qt.basis(4, 0)*qt.basis(4, 1).dag() + (Omega_c/2)*qt.basis(4, 1)*qt.basis(4, 0).dag() + \
            (Omega_p/2)*qt.basis(4, 0)*qt.basis(4, 2).dag() + (Omega_p/2)*qt.basis(4, 2)*qt.basis(4, 0).dag() + \
            (Omega_c/2)*qt.basis(4, 2)*qt.basis(4, 3).dag() + (Omega_c/2)*qt.basis(4, 3)*qt.basis(4, 2).dag() + \
            (Omega_p/2)*qt.basis(4, 1)*qt.basis(4, 3).dag() + (Omega_p/2)*qt.basis(4, 3)*qt.basis(4, 1).dag()  # Coupling terms

    H = term1 + term2  # Hamiltonian

    # Collapse operators
    A_cav = (qt.basis(4, 0)*qt.basis(4, 2).dag() + qt.basis(4, 1)*qt.basis(4, 3).dag()) * np.sqrt(kappa)
    A_relax = (qt.basis(4, 0)*qt.basis(4, 1).dag() + qt.basis(4, 2)*qt.basis(4, 3).dag()) * np.sqrt(Gamma)

    return H, [A_cav, A_relax]

# Function to calculate the spectrum of the EIT system
def sEIT_Spectrum(Omega_c, Omega_p, chi, kappa, Gamma, del_list, del_control, control="Probe"):
    """
    Calculate the spectrum of the EIT system.

    Parameters:
    Omega_c -- Rabi frequency of the control field
    Omega_p -- Rabi frequency of the probe field
    chi -- Dispersive shift
    kappa -- Cavity decay rate
    Gamma -- Relaxation rate
    del_list -- List of detuning values to sweep over
    del_control -- Detuning of the control field
    control -- Control parameter ("Probe" or "Couple")

    Returns:
    spec -- Spectrum of the EIT system
    """
    spec = np.zeros(len(del_list), dtype=np.float64)

    def spectrum(H, c_ops):
        rho_ss = qt.steadystate(H, c_ops, solver='scipy')
        return (rho_ss[1, 3] + rho_ss[0, 2]).imag

    if control == "Probe":
        for i, del_c in enumerate(del_list):
            H, c_ops = H_sEIT(del_c, del_control, Omega_c, Omega_p, chi, kappa, Gamma)
            spec[i] = spectrum(H, c_ops)
    else:
        for i, del_p in enumerate(del_list):
            H, c_ops = H_sEIT(del_control, del_p, Omega_c, Omega_p, chi, kappa, Gamma)
            spec[i] = spectrum(H, c_ops)
            
    return spec

# Parameters for the simulation
start = -10
end = 10
point = 201
chi = 2.5 * (2 * np.pi)  # Dispersive shift
del_control = 0
Omega_p = 0.1 * (2 * np.pi)
kappa = 1 * (2 * np.pi)
Gamma = 0.1 * (2 * np.pi)

# Detuning list
del_list = 2 * np.pi * np.linspace(start, end, point)
sweep_range = 1
resolution = 11

# Detuning values for control field
detune = [0.5, 1, 2]
z = np.zeros((resolution, point))

# Calculate and plot the spectrum
for i, Omega_c in enumerate(detune):
    z[i] = sEIT_Spectrum(Omega_c * (2 * np.pi), Omega_p, chi, kappa, Gamma, del_list, del_control, control="Couple")
    plt.plot(del_list / (2 * np.pi), z[i], label=f'Ω_c = {Omega_c}')

plt.xlabel("Cavity Detuning")
plt.ylabel("Imaginary Part of Spectrum")
plt.legend()
plt.grid()
plt.show()
