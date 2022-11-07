import plotly.graph_objects as go
import numpy as np

def plot_2D(xdata,ydata,zdata,xname,yname,zname):
    fig_2D = dict(
        layout = go.Layout(
            xaxis=dict(title = xname, showspikes = True, spikemode = 'across', tickformat = '.2e'),
            yaxis=dict(title = yname, showspikes = True, spikemode = 'across', tickformat = '.2e')),

        data = go.Heatmap(
            colorbar = dict(title = zname, tickformat = '.3e'),
            colorscale='Viridis', reversescale=True,
            x=xdata, y=ydata, z=zdata)
    )
    fig = go.Figure(fig_2D)
    fig.show()

def plot_1D(xdata,ydata,zdata,xname,yname,zname):
    fig_1D = dict(
        layout = go.Layout(
        xaxis=dict(title = xname, showspikes = True, spikemode = 'across', tickformat = '.2e'),
        yaxis=dict(title = zname, showspikes = True, spikemode = 'across', tickformat = '.2e')),
    )
    if np.shape(xdata) == np.shape(zdata):
        fig_1D["data"] = [go.Scatter(
            x = xdata, y = zdata,
            name = f'{yname} = {ydata}', mode='lines+markers')]
    else:
        fig_1D["data"] = [go.Scatter(
            x = xdata, y = data,
            name = f'{yname} ={ydata[i]}', mode='lines+markers') for i,data in enumerate(zdata)]
    
    fig = go.Figure(fig_1D)
    fig.show()

def plot_1Dslide(xdata,ydata,zdata,xname,yname,zname):
    fig = go.Figure(
        layout = go.Layout(
            xaxis=dict(
                title = xname, showspikes = True,
                spikemode = 'across', tickformat = '.2e'),
            yaxis=dict(
                title = zname, showspikes = True,
                spikemode = 'across', tickformat = '.2e'))
    )
    for i,data in enumerate(zdata):
        fig.add_trace(go.Scatter(visible=False, x = xdata, y = data, name = f'{yname} = {ydata[i]}'))
    fig.data[0].visible = True

    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],  # layout attribute
            label = f"{ydata[i]}"
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0, pad={"t": 50}, steps=steps,
        currentvalue={"prefix": f"{yname}: "})]

    fig.update_layout(sliders=sliders)

    fig.show()

def plot(xdata,ydata,zdata,xname,yname,zname):
    fig = go.Figure(
        layout = go.Layout(
            xaxis=dict(
                title = xname, showspikes = True,
                spikemode = 'across', tickformat = '.2e'),
            yaxis=dict(
                title = yname, showspikes = True,
                spikemode = 'across', tickformat = '.2e')),
    )
    if np.shape(xdata) == np.shape(zdata):
        fig.add_trace(go.Scatter(
            visible=False, x = xdata, y = zdata, name = f'{yname} = {ydata:.2f}'))
    else:
        for i,data in enumerate(zdata):
            fig.add_trace(go.Scatter(
                visible=False, x = xdata, y = data, name = f'{yname} = {ydata[i]:.2f}'))
    fig.add_trace(
        go.Heatmap(
            colorbar = dict(title = zname, tickformat = '.2e'),
            colorscale='Viridis', reversescale=True,
            x=xdata, y=ydata, z=zdata)
    )

    steps = []
    for i in range(len(ydata)):
        step = dict(
            method="update",
            args=[{"visible": ([False] * (len(ydata)+1))}],  # layout attribute
            label = f"{ydata[i]:.2e}"
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0, pad={"t": 50}, steps=steps,
        currentvalue={"prefix": f"{yname}: "})]


    fig.update_layout(
        updatemenus = [
            dict(
                type = 'buttons',
                direction = 'up',
                buttons = [
                    dict(
                        method = "update",
                        label = "2D Plot",
                        args = [
                            {"type":"heatmap",
                            "visible":[False]*len(ydata)+[True]},
                            {"yaxis":dict(title = yname, tickformat = '.2e')}]),
                    dict(
                        method = "update",
                        label = "1D Plot(All)",
                        args = [{
                            "type": "scatter",
                            "mode": "lines",
                            "visible":[True]*len(ydata)+[False]},
                            {"yaxis":dict(title = zname, tickformat = '.2e')}]),
                    dict(
                        method = "update",
                        label = "1D Plot",
                        args = [{
                            "type": "scatter",
                            "mode": 'lines+markers',
                            "visible":[True]+[False]*len(ydata)},
                            {"yaxis":dict(title = zname, tickformat = '.2e'),
                            "sliders": sliders}])
                ])
        ])
    fig.show()

