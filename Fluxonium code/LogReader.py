import numpy as np
import h5py
import pkg_resources
import sys
import plotly.graph_objects as go

_pkg_installed = [i.key for i in pkg_resources.working_set]

if 'labber' in _pkg_installed:
    import Labber
else:
    try:
        from .LabberAPI import Labber
    except ImportError:
        parent_dir = '../'
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
            from LabberAPI import Labber


def get_path(title: str = 'Select processing files'):
    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    paths = filedialog.askopenfilename(
        filetypes=[("Labber log files (*.hdf5)", "*.hdf5")],
        title=title)

    return paths


def switch_type(type, zdata):
    def unwrap_phase(z):
        return np.unwrap(np.angle(z))

    def origin(z):
        return z

    typedict = {
        'Magnitude': abs,
        'Real': np.real,
        'Imag': np.imag,
        'Phase': np.angle,
        '~Phase': unwrap_phase,
        'Complex': origin,
    }
    return typedict.get(type, abs)(zdata)


def is_concatenated_logs(fpath: str):
    with h5py.File(fpath, 'r') as f:
        f_dirnames = list(f.keys())
        return True if 'Log_2' in f_dirnames else False


def get_log_channel_name(logObj, idx: int):
    return logObj.getLogChannels()[idx]['name']


def get_instr_name(logObj):
    instr_name_tag_dict = {
        "Digitizer": ["Channel A", "Channel B"],
        "SA": ["Signal"],
        "VNA": [f'S{i}{j}' for i in range(1, 5) for j in range(1, 5)]
    }
    instr_tag = get_log_channel_name(logObj, -1).split(' - ')[1]
    instr_name = 'Undefined'
    for name_target, tag_list in instr_name_tag_dict.items():
        if instr_tag in tag_list:
            instr_name = name_target
    return instr_name


def get_data(logObj, path, instr_name):
    logchannel = logObj.getLogChannels()[0]
    channel1 = logObj.getStepChannels()[0]
    channel2 = logObj.getStepChannels()[1]

    loginfo = {
        'filename': path.split('/')[-1],
        'x': logObj.getTraceXY()[0],
        'y': np.array([0]),
        'z': logObj.getData(),
        'x_name': '', 'x_unit': '',
        'y_name': '', 'y_unit': '',
        'z_name': logchannel['name'],
        'z_unit': logchannel['unit'],
        'dim': 1 if np.shape(logObj.getData())[0] == 1 else 2
    }

    loginfo['instrument'] = instr_name

    if instr_name == 'VNA':
        loginfo.update(
            x_name='Frequency', x_unit='Hz',
            y=channel1['values'], y_name=channel1['name'], y_unit=channel1['unit']
        )
    elif instr_name == 'Digitizer':
        loginfo.update(x_name=channel1['name'], x_unit=channel1['unit'])
        if loginfo['dim'] != 1:
            loginfo.update(
                y=channel2['values'], y_name=channel2['name'], y_unit=channel2['unit']
            )

    if len(loginfo['y']) != np.shape(loginfo['z'])[0]:
        raise ValueError('y axis dimension is wrong')

    return loginfo


def parse_concatenated_hdf5(logObj, instr_name='VNA'):
    def get_freq_info(info):
        center = f'{instr_name} - Center frequency'
        span = f'{instr_name} - Span'
        start = f'{instr_name} - Start frequency'
        stop = f'{instr_name} - Stop frequency'

        if center in info.keys():
            f_start = info[center] - info[span] / 2
            f_stop = info[center] + info[span] / 2
        else:
            f_start = info[start]
            f_stop = info[stop]

        return f_start, f_stop

    log_ch_name = get_log_channel_name(logObj, 0)

    data = {
        'entries': logObj.getNumberOfEntries(),
        'x': None, 'x_name': 'Frequency', 'x_unit': 'Hz',
        'z': None, 'z_name': log_ch_name, 'z_unit': '',
        'y': [], 'y_name': '', 'y_unit': '',
    }

    entry_info = logObj.getEntry(entry=0)

    f_start, f_stop = get_freq_info(entry_info)
    for i in range(data['entries']):
        f_min, f_max = get_freq_info(logObj.getEntry(entry=i))
        if f_min < f_start:
            f_start = f_min
        if f_max > f_stop:
            f_stop = f_max
    f_delta = entry_info[log_ch_name]['dt']
    f_tot_pts = int((f_stop - f_start) / f_delta)
    data['x'] = np.linspace(start=f_start, stop=f_stop, num=f_tot_pts + 1)

    ydict = logObj.getStepChannels()[-1]
    data['y_name'], data['y_unit'] = ydict['name'], ydict['unit']
    ylist = [(i, logObj.getEntry(entry=i)[data['y_name']])
             for i in range(data['entries'])]
    ylist.sort(key=lambda s: s[1])

    data['z'] = np.full(
        (data['entries'], int(len(data['x']))), np.nan, np.cfloat)

    for i, flux in enumerate(ylist):
        idx = flux[0]
        entry_info = logObj.getEntry(entry=idx)
        start_freq, stop_freq = get_freq_info(entry_info)
        start_idx, = np.where(data['x'] == start_freq)
        stop_idx, = np.where(data['x'] == stop_freq)
        data['z'][i][start_idx[0]:stop_idx[0] + 1] = entry_info[
            get_log_channel_name(logObj, 0)]['y']

        data['y'].append(entry_info[data['y_name']])

    data['y'] = np.array(data['y'])

    return data


def get_tag(logObj) -> dict:
    info_dict = {}
    project = logObj.getProject().split('/')
    sample, projectname = '', ''
    sample = project[0]
    if len(project) > 1:
        projectname = project[1]
    info_dict["sample"] = sample
    info_dict["proj"] = projectname
    info_dict["user"] = logObj.getUser()
    info_dict["tag"] = logObj.getTags()
    return info_dict


def plot_1D(xdata, ydata, zdata, xname, yname, zname):
    fig_1D = dict(
        layout=go.Layout(
            xaxis=dict(title=xname, showspikes=True,
                       spikemode='across', tickformat='.2e'),
            yaxis=dict(title=zname, showspikes=True, spikemode='across', tickformat='.2e')),
    )
    if np.shape(xdata) == np.shape(zdata):
        fig_1D["data"] = [go.Scatter(
            x=xdata, y=zdata,
            name=f'{yname} = {ydata}', mode='lines+markers')]
    else:
        fig_1D["data"] = [go.Scatter(
            x=xdata, y=data,
            name=f'{yname} ={ydata[i]}', mode='lines+markers') for i, data in enumerate(zdata)]

    fig = go.Figure(fig_1D)
    fig.show()


def plot(xdata, ydata, zdata, xname, yname, zname):
    fig = go.Figure(
        layout=go.Layout(
            xaxis=dict(
                title=xname, showspikes=True,
                spikemode='across', tickformat='.2e'),
            yaxis=dict(
                title=yname, showspikes=True,
                spikemode='across', tickformat='.2e')),
    )
    if np.shape(xdata) == np.shape(zdata):
        fig.add_trace(go.Scatter(
            visible=False, x=xdata, y=zdata, name=f'{yname} = {ydata:.2f}'))
    else:
        for i, data in enumerate(zdata):
            fig.add_trace(go.Scatter(
                visible=False, x=xdata, y=data, name=f'{yname} = {ydata[i]:.2f}'))
    fig.add_trace(
        go.Heatmap(
            colorbar=dict(title=zname, tickformat='.2e'),
            colorscale='Viridis', reversescale=True,
            x=xdata, y=ydata, z=zdata)
    )

    steps = []
    for i in range(len(ydata)):
        step = dict(
            method="update",
            args=[{"visible": ([False] * (len(ydata) + 1))}],
            label=f"{ydata[i]:.2e}"
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)

    sliders = [dict(
        active=0, pad={"t": 50}, steps=steps,
        currentvalue={"prefix": f"{yname}: "})]

    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                direction='up',
                buttons=[
                    dict(
                        method="update", label="2D Plot",
                        args=[
                            {"type": "heatmap", "visible": [False] * len(ydata) + [True]},
                            {"yaxis": dict(title=yname, tickformat='.2e')}]),
                    dict(
                        method="update", label="1D Plot(All)",
                        args=[
                            {"type": "scatter", "mode": "lines", "visible": [True] * len(ydata) + [False]},
                            {"yaxis": dict(title=zname, tickformat='.2e')}]),
                    dict(
                        method="update", label="1D Plot",
                        args=[
                            {"type": "scatter", "mode": 'lines+markers', "visible": [True] + [False] * len(ydata)},
                            {"yaxis": dict(title=zname, tickformat='.2e'), "sliders": sliders}])
                ])
        ])
    fig.show()


class LogHandler:
    """
    Data extractor

    Parameters
    ----------
    file : str
        Labber log file path

    Instance attributes: 
    --------------------
        loginfo:
            filename: str = filename
            instrument: str = measurement instrument
            dim: int = dimension of z data

            x: np.ndarray = xdata
            y: np.ndarray = ydata
            z: np.ndarray = zdata

            x_name: str = x axis name 
            y_name: str = y axis name 
            z_name: str = z axis name 

            x_unit: str = x axis unit 
            y_unit: str = y axis unit 
            z_unit: str = z axis unit 

            sample: str = sample name
            proj: str = projectname
            user: str = User
            tag: str = Tags
    
    Methods:
    --------
        (see output help for detail)
        output
        plot
    """

    def __init__(self, file=None):
        self.path = file if file is not None else get_path()
        self.log = Labber.LogFile(self.path)
        self.instr_name = get_instr_name(self.log)
        if is_concatenated_logs(self.path) and self.instr_name == 'VNA':
            self.loginfo = parse_concatenated_hdf5(self.log, self.instr_name)
        else:
            self.loginfo = get_data(self.log, self.path, self.instr_name)
        self.loginfo.update(get_tag(self.log))

    def output(self, data_type: str = 'Magnitude', entry=None, avg=False):
        """
        Output data

        Parameters
        ----------
        data_type : str
            z data type:
                'Magnitude': Magnitude
                'Real': Real part
                'Imag': Imaginary part
                'Phase': Phase
                '~Phase': Unwrap Phase
                'Complex': Complex
        entry : int, tuple, list, None
            int: output the specific entry
            tuple: (start, stop) output a range of entry
            list: output the selected entry
            None: output whole data
        avg : bool, optional
            average all the selected data

        Returns
        -------
        x: np.ndarray = x data
        y: np.ndarray = y data
        z: np.ndarray = z data
        xname: str = x axis name with units
        yname: str = y axis name with units
        zname: str = z axis name with units
        """

        xname = f'{self.loginfo["x_name"]} [{self.loginfo["x_unit"]}]'
        yname = f'{self.loginfo["y_name"]} [{self.loginfo["y_unit"]}]'
        zname = f'{self.loginfo["z_name"]} [{self.loginfo["z_unit"]}]'
        x = self.loginfo['x']
        y = self.loginfo['y']
        z = self.loginfo['z']

        if entry is not None:
            if isinstance(entry, int):
                y = np.array(y[entry])
                z = np.array(z[entry])
            elif isinstance(entry, tuple):
                y = y[entry[0] - 1: entry[1]]
                z = z[entry[0] - 1: entry[1]]
            elif isinstance(entry, (list, np.ndarray)):
                newlist = np.sort(np.array(entry) - 1)
                y = np.array([y[i] for i in newlist])
                z = np.array([z[i] for i in newlist])
        z = switch_type(data_type, z)
        if avg:
            z = np.sum(z, 0) / len(y)
        return x, y, z, xname, yname, zname

    def plot(self, data_type='Magnitude', entry=None, avg=False):
        """
        Plot data

        Parameters
        ----------
        data_type : str, optional
        entry : int, tuple, list, None
        avg : bool

        Returns
        -------
        plot
        """
        arg = self.output(data_type, entry, avg)
        if len(self.loginfo['y']) == 1 or isinstance(entry, int):
            plot_1D(*arg)
        else:
            plot(*arg)


if __name__ == '__main__':
    path = get_path()
    testlog = LogHandler(path)
    testlog.plot()
