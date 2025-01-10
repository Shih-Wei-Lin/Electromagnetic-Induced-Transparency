
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
from numpy import *
import numpy as np
import scqubits as scq
from matplotlib import pyplot as plt
from labellines import labelLines
import h5py
from hdf5Reader  import *
from LogReader  import *
from plotly import *
import plotly.graph_objects as go
import seaborn as sns

def plot_transition_spectrum_data(fileDict,fluxonium,evals_count,point,zeroc,halfc,Phase=False):
    def current(c):
            phi=c*(0.5/(halfc-zeroc))-zeroc*(0.5/(halfc-zeroc))
            return phi
    
#     #import data from labber(use VNA)

    testlog =LogHandler(fileDict)
    data = testlog.loginfo
    
    if Phase:
        output_exp=testlog.output("~Phase")[2]
    else:
        output_exp=testlog.output("Magnitude")[2]
    
    
    data["y"]= current(data["y"]/10**-6)
    data["x"]=data["x"]/10**9

    flux_list = np.linspace(min(data["y"]),max(data["y"]), 301)
    eigval = fluxonium.get_spectrum_vs_paramvals("flux", flux_list, evals_count).energy_table
    eigval_list = eigval.T
    N = np.shape(eigval_list)[0]
    # probable transition:
    # 0-f, 1-f
    spectrum_list_0 = np.zeros((N-1, len(flux_list)))
    spectrum_list_1 = np.zeros((N-2, len(flux_list)))
    for i in range(N-1):
         spectrum_list_0[i] = eigval_list[i+1] - eigval_list[0]
    for i in range(N-2):
         spectrum_list_1[i] = eigval_list[i+2] - eigval_list[1]




    fig = go.Figure(
        layout = go.Layout(
                xaxis=dict(title = r"$\frac{\phi_{ext}}{2\pi}$", showspikes = True, spikemode = 'across'),
                yaxis=dict(title = "Frequency(GHz)", showspikes = True, spikemode = 'across'),title="Transition spectrum",
                     width=800, height=600),
                
        data=go.Heatmap(
                showscale = False,
                x=data["y"], y=data["x"], z=output_exp.T,colorscale='Greys', reversescale=True))#Viridis RdBu_r
    for idx in range(np.shape(spectrum_list_0)[0]):
        fig.add_traces([go.Scatter(x = flux_list, y = (spectrum_list_0[idx]),name=f"{idx+1},0",line=dict(dash="longdash"))] )# 0-f
    #     if idx !=0:
    #         fig.add_traces([go.Scatter(x = flux_list, y = (spectrum_list_0[idx]/2),name=f"{idx+1},0 2p")] )#Two photon0-f
    for idx in range(np.shape(spectrum_list_1)[0]):
        fig.add_traces([go.Scatter(x = flux_list, y = (spectrum_list_1[idx]),name=f"{idx+2},1",line=dict(dash="longdash"))] )#1-f
        #fig.add_traces([go.Scatter(x = flux_list, y = (spectrum_list_1[idx]/2),name=f"{idx+2},1 2p")] )#Two photon1-f

    fig.update_yaxes(range=list([min(data["x"])-0.2,max(data["x"])+0.2]))
    fig.show()

    fig,ax=fluxonium.plot_matelem_vs_paramvals('n_operator', 'flux', flux_list, select_elems=[(0, 1),(0,2),(1,2)])
    fig.set_size_inches(8.5, 6)
    plt.xlim([ min(data["y"]),max(data["y"])])
    plt.ylabel("Matrixelement", fontsize=30)

# %matplotlib notebook
fluxonium = scq.Fluxonium(
    EJ=7,
    EC=0.8,
    EL=1.17,
    cutoff = 50,
    flux = 0.5
)
#     EJ=6.305,
#     EC=0.77,
#     EL=0.94,
fileDict = r'C:\Users\user\SynologyDrive\DESKTOP-V6JLHQU\C\Users\cluster\Labber\Data\2022\10\Data_1028\Fluxonium043_RBProtocol_two_tone_sweep_flux_002.hdf5'#input file path(only for VNA)#input file path(only for VNA)
evals_count=3
plot_transition_spectrum_data(fileDict,fluxonium,evals_count,301,-757,698.61,Phase=False) #(file path,fluxonium,evals_count,flux resolution,half quanta current(uA),integer quanta)
