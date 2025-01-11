import numpy as np
import scqubits as scq
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from LogReader import LogHandler

def plot_transition_spectrum_data(fileDict, fluxonium, evals_count, point, zeroc, halfc, Phase=False):
    def current(c):
        phi = c * (0.5 / (halfc - zeroc)) - zeroc * (0.5 / (halfc - zeroc))
        return phi

    # Import data from Labber (use VNA)
    testlog = LogHandler(fileDict)
    data = testlog.loginfo
    
    if Phase:
        output_exp = testlog.output("~Phase")[2]
    else:
        output_exp = testlog.output("Magnitude")[2]
    
    data["y"] = current(data["y"] / 10**-6)
    data["x"] = data["x"] / 10**9

    flux_list = np.linspace(min(data["y"]), max(data["y"]), 301)
    eigval = fluxonium.get_spectrum_vs_paramvals("flux", flux_list, evals_count).energy_table
    eigval_list = eigval.T
    N = np.shape(eigval_list)[0]

    spectrum_list_0 = np.zeros((N-1, len(flux_list)))
    spectrum_list_1 = np.zeros((N-2, len(flux_list)))
    for i in range(N-1):
        spectrum_list_0[i] = eigval_list[i+1] - eigval_list[0]
    for i in range(N-2):
        spectrum_list_1[i] = eigval_list[i+2] - eigval_list[1]

    fig = go.Figure(
        layout = go.Layout(
            xaxis=dict(title=r"$\frac{\phi_{ext}}{2\pi}$", showspikes=True, spikemode='across'),
            yaxis=dict(title="Frequency (GHz)", showspikes=True, spikemode='across'),
            title="Transition Spectrum",
            width=800,
            height=600
        ),
        data=go.Heatmap(
            showscale=False,
            x=data["y"],
            y=data["x"],
            z=output_exp.T,
            colorscale='Greys',
            reversescale=True
        )
    )

    for idx in range(np.shape(spectrum_list_0)[0]):
        fig.add_traces([go.Scatter(x=flux_list, y=spectrum_list_0[idx], name=f"{idx+1},0", line=dict(dash="longdash"))])
    for idx in range(np.shape(spectrum_list_1)[0]):
        fig.add_traces([go.Scatter(x=flux_list, y=spectrum_list_1[idx], name=f"{idx+2},1", line=dict(dash="longdash"))])

    fig.update_yaxes(range=[min(data["x"])-0.2, max(data["x"])+0.2])
    fig.show()

    fig, ax = fluxonium.plot_matelem_vs_paramvals('n_operator', 'flux', flux_list, select_elems=[(0, 1), (0, 2), (1, 2)])
    fig.set_size_inches(8.5, 6)
    plt.xlim([min(data["y"]), max(data["y"])])
    plt.ylabel("Matrix Element", fontsize=30)

fluxonium = scq.Fluxonium(
    EJ=7,
    EC=0.8,
    EL=1.17,
    cutoff=50,
    flux=0.5
)

fileDict = r'C:\Users\user\SynologyDrive\DESKTOP-V6JLHQU\C\Users\cluster\Labber\Data\2022\10\Data_1028\Fluxonium043_RBProtocol_two_tone_sweep_flux_002.hdf5'
evals_count = 3
plot_transition_spectrum_data(fileDict, fluxonium, evals_count, 301, -757, 698.61, Phase=False)
