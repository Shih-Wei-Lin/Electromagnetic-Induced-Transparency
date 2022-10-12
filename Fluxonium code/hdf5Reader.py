# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 15:52:32 2021

@author: Alaster
"""

import numpy as np
from tkinter import filedialog, Tk
import h5py
from collections import Counter
import matplotlib.pyplot as plt
from shutil import copyfile
from pandas import read_csv, DataFrame
from os import rename
from os.path import join, split, splitext


# General utilities
def get_path(ext='.csv', title: str = 'Select item'):
    """
    Interactive dialogue box to get file for a certain extension.

    Parameters
    ----------
    title : str, optional
        Title of the dialogue box. The default is 'Select item'.

    Returns
    -------
    str
        Path string.

    """
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return filedialog.askopenfilename(
        filetypes=[(ext.upper() + ' Files', ext)], title=title
        )


def get_hdf5_path(title: str = 'Select item'):
    """
    Interactive dialogue box to get hdf5 file.

    Parameters
    ----------
    title : str, optional
        Title of the dialogue box. The default is 'Select item'.

    Returns
    -------
    str
        Path string.

    """
    return get_path('.hdf5', title)


def plot(x: np.array, y: np.array, z: np.array,
         xname: str = 'xname', yname: str = 'yname', zname: str = 'zname'):
    """
    Plotting Interface. The default is 2D-plot if y-axis has only 1
    entry, otherwise auto-switch to contour plot.

    Parameters
    ----------
    x : np.array
        x-axis array.
    y : np.array
        y-axis array.
    z : np.array
        z-axis mesh array.
    xname : str, optional
        x-axis title. The default is 'xname'.
    yname : str, optional
        y-axis title. The default is 'yname'.
    zname : str, optional
        z-axis title. The default is 'zname'.

    """
    if len(y) > 1:
        contour(x, y, z, xname, yname, zname)
        return
    fig, ax = plt.subplots()
    ax.set_title(yname + ' = ' + str(y[0]))
    ax.plot(x, np.abs(z), 'b')
    ax.set_xlabel(xname)
    ax.set_ylabel(zname + ' Magnitude', color="blue")
    ax2 = ax.twinx()
    ax2.plot(x, np.angle(z, True), 'r')
    ax2.set_ylabel(zname + ' Phase (deg)', color="red")


def contour(x: np.array, y: np.array, z: np.array,
            xname: str = 'xname', yname: str = 'yname', zname: str = 'zname'):
    """
    Plotting contour map.

    Parameters
    ----------
    x : np.array
        x-axis array.
    y : np.array
        y-axis array.
    z : np.array
        z-axis mesh array.
    xname : str, optional
        x-axis title. The default is 'xname'.
    yname : str, optional
        y-axis title. The default is 'yname'.
    zname : str, optional
        z-axis title. The default is 'zname'.

    """
    if np.iscomplexobj(z):
        z = np.abs(z)
    a = plt.pcolormesh(*np.meshgrid(x, y), z.transpose(), shading='auto')
    plt.xlabel(xname)
    plt.ylabel(yname)
    b = plt.colorbar(a)
    b.ax.set_ylabel(zname)
    plt.show()


def zoom(x: np.array, y: np.array, z: np.array,
         xRange: tuple = (-np.inf, np.inf), yRange: tuple = (-np.inf, np.inf)):
    """
    Return zoomed x, y, z arrays.

    Parameters
    ----------
    x : np.array
        x-axis array.
    y : np.array
        y-axis array.
    z : np.array
        z-axis mesh array.
    xRange : tuple, optional
        Range values for x-axis. The default is (-np.inf, np.inf).
    yRange : tuple, optional
        Range values for y-axis. The default is (-np.inf, np.inf).

    Returns
    -------
    x_zoom : np.array
        zoomed x-axis.
    y_zoom : np.array
        zoomed y-axis.
    z_zoom : np.array
        zoomed z-axis.

    """
    # zoom x
    index_bool = x <= xRange[1]
    x_zoom, z_zoom = x[index_bool], z[index_bool, :]
    index_bool = x_zoom >= xRange[0]
    x_zoom, z_zoom = x_zoom[index_bool], z_zoom[index_bool, :]

    # zoom y
    index_bool = y <= yRange[1]
    y_zoom, z_zoom = y[index_bool], z_zoom[:, index_bool]
    index_bool = y_zoom >= yRange[0]
    y_zoom, z_zoom = y_zoom[index_bool], z_zoom[:, index_bool]
    return x_zoom, y_zoom, z_zoom


def selectFromList(listItem: list, qText: str = '',
                   returnIndex: bool = False, allOption: bool = False):
    """
    Return the selected object or index from a list.

    Parameters
    ----------
    listItem : list
        List of objects.
    qText : str, optional
        Dialogue text. The default is ''.
    returnIndex : bool, optional
        Set True to return index, otherwise selected item from the list. The
        default is False.
    allOption : bool, optional
        Set True to enable 'all' selection keyword ':'. Enter non-int text
        after the dialogue text to trigger all item output. The default is
        False.

    """
    itemString = ''
    for i, j in enumerate(listItem):
        itemString += str(i) + ': ' + str(j) + '\n'
    try:
        val = input(qText + ':\n' + itemString + '=>')
        idx = int(val)
    except(TypeError):
        if allOption:
            return ':'
        raise ValueError(
            'The input field must be an integer unless allOption is enabled'
            )
    if returnIndex:
        return idx
    return listItem[idx]


def formatConvert(z: np.array, yshape: list = []):
    """
    Conversion between Labber format and common Fortran type array shape. Empty
    yshape argument leads to conversion to Labber format or inverted operation
    otherwise.

    Parameters
    ----------
    z : np.array
        z-axis data to be converted.
    yshape : list, optional
        Shape data for the final output. The default is [].

    Returns
    -------
    np.array
        Transformed z-axis.

    """
    if yshape:  # Labber hdf5 format to normal format
        row, col, layer = z.shape

        # if yshape is incomplete due to interrupt during measurement. Use
        # nan to fill up the empty data.
        if layer != yshape[0]:
            nan_array = np.nan * np.zeros((row, col, yshape[0]))
            nan_array[:, :, :layer] = z[:]
            z = nan_array
        ztemp = z[:, 0, :]

        if z.shape[1] > 1:
            ztemp = ztemp + 1j * z[:, 1, :]
        return np.reshape(ztemp, [ztemp.shape[0]] + list(yshape), 'F')

    # normal format to Labber hdf5 format
    if np.iscomplexobj(z):
        ztemp = np.reshape(z, [z.shape[0], np.prod(z.shape[1:])], 'F')
        ztemp = np.stack((np.real(ztemp), np.imag(ztemp)), axis=1)
    else:
        ztemp = np.reshape(z, [z.shape[0], 1, np.prod(z.shape[1:])], 'F')
    return ztemp


def valueSelection(axis: np.array, name: str, unit: str,
                   qText: str = 'Select case from below'):
    """
    Select a value from an array. This function returns index corresponds to
    the value closest to the input from the given array.

    Parameters
    ----------
    axis : np.array
        Array used to select value.
    name : str
        Name of the array.
    unit : str
        Unit of the array data.
    qText : str, optional
        Descirption of the array. The default is 'Select case from below'.

    Returns
    -------
    idx : int
        Selected index.

    """
    print('\n' + qText)
    print('Axis name: ' + name + ' (' + unit + ')')
    step = axis[1] - axis[0] if len(axis) > 1 else 0
    print(
        f'max= {max(axis)}, min: {min(axis)}, step: ' + '{:.4e}'.format(step)
        )
    flag = True
    while flag:
        val = input(f'Enter value {unit}=>')
        try:
            val = float(val)
            flag = False
        except ValueError:
            continue
    idx = (np.abs(axis - val)).argmin()
    return idx


def csv2dat(csvfile):
    """
    .csv to .dat file conversion.

    Parameters
    ----------
    csvfile : str
        Input filename.

    Returns
    -------
    dst : str
        Output filename.

    """
    data = read_csv(csvfile, header=None)
    data.to_csv(csvfile, mode='w', sep='\t', index=False, header=False)
    dst = splitext(csvfile)[0] + '.dat'
    rename(csvfile, dst)
    return dst


def dat2csv(datfile):
    """
    .dat to .csv file conversion.

    Parameters
    ----------
    datfile : str
        Input filename.

    Returns
    -------
    dst : str
        Output filename.

    """
    data = read_csv(datfile, header=None, sep='\t')
    data.to_csv(datfile, mode='w', sep=',', index=False, header=False)
    dst = splitext(datfile)[0] + '.csv'
    rename(datfile, dst)
    return dst


def get_zname(filename, log_ch_idx=0):
    """
    Get the z-axis name of the instrument.

    Parameters
    ----------
    filename : str
        HDF fimename.
    log_ch_idx : int, optional
        Index of desired log channel. The default is 0.

    Returns
    -------
    zname : str
        z-axis name.

    """
    f = h5py.File(filename, 'r')
    zname = f['Log list'][log_ch_idx][0]
    return zname


def get_zInstru_name(filename, log_ch_idx=0):
    """
    Get the name of the instrument.

    Parameters
    ----------
    filename : str
        HDF fimename.
    log_ch_idx : int, optional
        Index of desired log channel. The default is 0.

    Returns
    -------
    str
        Instrument name.

    """
    zname = get_zname(filename, log_ch_idx)
    return zname.split(' - ')[0]


#  Backend functions
def _selectyaxis(axisList, namelist, unitlist):
    flag = selectFromList(namelist, 'Select output y-axis', True, True)
    if isinstance(flag, str):   # select all axis as output
        dimList = [range(len(item)) for item in axisList]
        return axisList, namelist, unitlist, dimList
    dimList = []
    for i, (axis, name, unit) in enumerate(zip(axisList, namelist, unitlist)):
        if i == flag:   # range all value for selected y axis
            dimList += [range(len(axis))]
            continue
        idx = valueSelection(axis, name, unit)
        dimList += [idx]
    return axisList[flag], namelist[flag], unitlist[flag], dimList


def _meshSlice(zMesh, yaxisList, ynamelist, yunitlist):
    if len(ynamelist) > 1:
        y, yname, yunit, dimList = _selectyaxis(
            yaxisList, ynamelist, yunitlist
            )
        temp = np.moveaxis(zMesh, 0, -1)[tuple(dimList)]
        z = np.moveaxis(temp, -1, 0)
    else:
        y, yname, yunit, dimList = \
            yaxisList[0], ynamelist[0], yunitlist[0], [range(0)]
        z = zMesh

    # create record for sliced values
    record = [[f'sliceRec{col}', name, unit, str(axis[idx])] for col, (
        name, unit, axis, idx) in enumerate(
            zip(ynamelist, yunitlist, yaxisList, dimList)
        ) if not isinstance(idx, range)]
    recordT = np.array(record, dtype=object).T
    return z, y, yname, yunit, recordT


def _sort_Data_ys(f):
    nameList, axisList, dimList = [], [], []

    # return empty dict and list if nothing is in 'Channel names'
    if len(f['Data/Channel names'][:]) == 0:
        return [''], [np.array([0])], [1]

    # use total points to estimate number of indepedent variables
    def_name = _tostr(f['Log list'][0][0])
    total_DOF = len(f['Traces'][def_name][0, 0, :])
    i = 0
    while total_DOF > 1:  # escape if num of independent variable is reached
        nameList += [_tostr(f['Data/Channel names'][:][i][0])]
        val = f['Data/Data'][:, i, :]
        if i == 0:  # mainloop
            axisList += [val[:, 0]]
        else:   # other layer of loops
            axisList += [np.sort(list(Counter(val[0, :]).keys()))]
        dimVal = len(axisList[i])
        dimList += [dimVal]
        total_DOF //= dimVal
        i += 1
    return nameList, axisList, dimList


def _sort_Traces_xz(f, log_ch_idx=None, yshape=[]):
    if len(f['Log list']) == 1:
        log_ch_idx = 0
    if log_ch_idx is None:
        namelist = [
            _tostr(f['Log list'][i][0]) for i in range(len(f['Log list']))
            ]
        log_ch_idx = selectFromList(
            namelist, 'Choose the log channel index from below', True
            )
    log_name = _tostr(f['Log list'][log_ch_idx][0])
    data = f['Traces']
    t0, dt = data[log_name + '_t0dt'][:][0, :]
    N = data[log_name + '_N'][:][0]
    x = t0 + dt * np.linspace(0, N, N+1)[:-1]
    z = formatConvert(data[log_name], yshape)
    return x, z, log_name


def _get_unit(f, channel_name, returnIdx=False):
    data = f['Channels']
    idx = [_tostr(data[i][0]) for i in range(len(data))].index(channel_name)
    if returnIdx:
        return idx
    unit = _tostr(data[idx][3])
    return unit


def _tostr(strObj):
    if isinstance(strObj, bytes):
        return strObj.decode("utf-8")
    if isinstance(strObj, str):
        return strObj
    else:
        raise TypeError('Incorrect datatype')


# Interfaces
def open_hdf5(file: str = None, log_ch_idx: int = None):
    """
    Open and extract Labber-formatted hdf5 file.

    Parameters
    ----------
    file : str, optional
        Path string of the hdf5 file. The default is None.
    log_ch_idx : int, optional
        Log channel index. This depends on the Labber measurement file setting.
        The default is None.

    Returns
    -------
    x : np.array
        x-axis.
    yaxisList : list
        List of potential y-axis.
    z : np.array
        z-axis.
    ynamelist : list
        Name list of potential y-axis.
    zname : str
        Name of z-axis.
    yunitlist : list
        Unit list of y-axis.
    zunit : str
        Unit of z-axis.

    """
    if not file:
        file = get_path('.hdf5')
    with h5py.File(file, 'r') as f:
        # read y axis data
        ynamelist, yaxisList, dimList = _sort_Data_ys(f)
        # read x, z axis data
        x, z, zname = _sort_Traces_xz(f, log_ch_idx, dimList)
        yunitlist = [
            _get_unit(f, yname) if yname else '' for yname in ynamelist
            ]
        zunit = _get_unit(f, zname)
    return x, yaxisList, z, ynamelist, zname, yunitlist, zunit


def get_VNA_Data(file: str = None, bypass: bool = False,
                 bg_row: int = None, collapse_yz: bool = False,
                 log_ch_idx: int = None
                 ):
    """
    Get VNA data and slice into a Fortran-style data array.

    Parameters
    ----------
    file : str, optional
        Path string of the hdf5 file. The default is None.
    bypass : bool, optional
        Set True to load the data from hdf5 without unfolding operation. The
        default is False.
    bg_row : int, optional
        Background row index for VNA data format. The default is None.
    collapse_yz : bool, optional
        Set True to set the output is rank-2, otherwise rank-3. The default is
        False.
    log_ch_idx : int, optional
        Log channel index. This depends on the Labber measurement file setting.
        The default is None.

    Returns
    -------
    x : np.array
        x-axis.
    list or np.array
        list of y-axis or single y-axis.
    np.array
        Unsliced or sliced z-axis.
    xname : str
        x-axis name.
    list or str
        list of y-axis name or single y-axis name.
    zname : str
        z-axis name.

    """
    data = hdf5Handle('Frequency', 'Hz', file, log_ch_idx)
    if bypass:
        return data.output(raw=True)
    data.slice()
    x, y, z, xname, yname, zname, record = data.output()
    if bg_row is not None:
        if collapse_yz:
            y, z = y[bg_row], z[:, bg_row]
        else:
            y, z = y[bg_row, np.newaxis], z[:, bg_row, np.newaxis]
    return x, y, z, xname, yname, zname, record


def get_SA_Data(file: str = None, bypass: bool = False,
                bg_row: int = None, collapse_yz: bool = False):
    """
    Get SA data and slice into a Fortran-style data array.

    Parameters
    ----------
    file : str, optional
        Path string of the hdf5 file. The default is None.
    bypass : bool, optional
        Set True to load the data from hdf5 without unfolding operation. The
        default is False.
    bg_row : int, optional
        Background row index for SA data format. The default is None.
    collapse_yz : bool, optional
        Set True to set the output is rank-2, otherwise rank-3. The default is
        False.

    Returns
    -------
    x : np.array
        x-axis.
    list or np.array
        list of y-axis or single y-axis.
    np.array
        Unsliced or sliced z-axis.
    xname : str
        x-axis name.
    list or str
        list of y-axis name or single y-axis name.
    zname : str
        z-axis name.

    """
    data = hdf5Handle('Frequency', 'Hz', file)
    data.zunit = 'dBm'
    if bypass:
        return data.output(raw=True)
    data.slice()
    x, y, z, xname, yname, zname, record = data.output()
    if bg_row is not None:
        if collapse_yz:
            y, z = y[bg_row], z[:, bg_row]
        else:
            y, z = y[bg_row, np.newaxis], z[:, bg_row, np.newaxis]
    return x, y, z, xname, yname, zname, record


def get_Digitizer_data(file: str = None, bypass: bool = False):
    """
    Get digitizer data and slice into a Fortran-style data array.

    Parameters
    ----------
    file : str, optional
        Path string of the hdf5 file. The default is None.
    bypass : bool, optional
        Set True to load the data from hdf5 without unfolding operation. The
        default is False.

    Returns
    -------
    x : np.array
        x-axis.
    list or np.array
        list of y-axis or single y-axis.
    np.array
        Unsliced or sliced z-axis.
    xname : str
        x-axis name.
    list or str
        list of y-axis name or single y-axis name.
    zname : str
        z-axis name.

    """
    data = hdf5Handle('Time', 's', file)
    if bypass:
        return data.output(raw=True)
    data.slice()
    return data.output()


# file management
def copy_hdf5(file: str, prefix: str = '', suffix: str = ''):
    """
    Copy a hdf5 file.

    Parameters
    ----------
    file : str
        File to be copied.
    prefix : str, optional
        Filename prefix. The default is ''.
    suffix : str, optional
        Filename suffix. The default is ''.

    Returns
    -------
    sv_file : str
        Name of the newly created file.

    """
    # copy data/setup to destination
    directory, filename = split(file)
    filename, extension = splitext(filename)
    sv_file = join(directory, prefix + filename + suffix + extension)
    copyfile(file, sv_file)
    return sv_file


def add_to_hdf5(data: np.array, channel: str, file: str,
                unit: str = None, prefix: str = '', suffix: str = ''):
    """
    Add log channel data to the existed hdf5 file.

    Parameters
    ----------
    data : np.array
        Data to be saved.
    channel : str
        Name of the new log channel.
    file : str
        Destination file name.
    unit : str, optional
        Unit of the log channel. The default is None.
    prefix : str, optional
        Prefix for modified log channel name. The default is ''.
    suffix : str, optional
        Suffix for modified log channel name. The default is ''.

    """
    new_ch = prefix + channel + suffix

    with h5py.File(file, 'r+') as f:
        # get reference name
        ref_ch = _tostr(f['Log list'][0][0])

        # replace Log list
        new_array = np.insert(f['Log list'][:], -1, new_ch)
        del f['Log list']
        f.create_dataset('Log list', data=new_array)

        # Add new Trace properties
        t = f['Traces']
        t.create_dataset(new_ch, data=data)
        t.create_dataset(new_ch + '_N', data=t[ref_ch + '_N'][:])
        t.create_dataset(new_ch + '_t0dt', data=t[ref_ch + '_t0dt'][:])

        # Add new Channels
        ch_duplicate = f['Channels'][_get_unit(f, ref_ch, True)]
        ch_duplicate[0] = new_ch
        if unit is not None:
            ch_duplicate[3] = unit
            ch_duplicate[4] = unit
        new_array = np.insert(f['Channels'][:], -1, ch_duplicate)
        del f['Channels']
        f.create_dataset('Channels', data=new_array)


class hdf5Handle():

    def __init__(self, xname='', xunit='', file=None, log_ch_idx=None):
        self.xname = xname
        self.xunit = xunit
        self.x, self.yaxisList, self.zMesh, self.ynamelist, self.zname,\
            self.yunitlist, self.zunit = open_hdf5(file, log_ch_idx)

    def slice(self):
        self.z, self.y, self.yname, self.yunit, self.record = _meshSlice(
            self.zMesh, self.yaxisList, self.ynamelist, self.yunitlist
            )

    def plot(self):
        if not hasattr(self, 'yname'):
            self.slice()
        plot(self.x, self.y, self.z, self.xname, self.yname, self.zname)

    def output(self, raw=False):
        xname = self.xname + ' (' + self.xunit + ')'
        zname = self.zname + ' (' + self.zunit + ')'

        if raw:
            ynamelist = [
                yname + ' (' + yunit + ')' for yname, yunit in zip(
                    self.ynamelist, self.yunitlist)
                ]
            return self.x, self.yaxisList, self.zMesh, xname, ynamelist, \
                zname, []
        else:
            if not hasattr(self, 'yname'):
                self.slice()
            yname = self.yname + ' (' + self.yunit + ')'
            return self.x, self.y, self.z, xname, yname, zname, self.record

    def to_file(self, filename='', ext='.csv', delimiter=',', cplxSplit=True):
        if not hasattr(self, 'yname'):
            self.slice()
        if not filename:
            filename = self.zname

        # x, y axis information
        file = splitext(filename)[0]
        fileList = [file + '_axisInfo' + ext]

        dy = 0
        if len(self.y) > 1:
            self.y[1] - self.y[0]

        axisInfo = [
            ['X', 'Y'],
            [self.xname, self.yname],
            [self.xunit, self.yunit],
            [self.x[0], self.y[0]],
            [self.x[-1], self.y[-1]],
            [self.x[1] - self.x[0], dy]
            ]
        DataFrame(axisInfo).to_csv(
            fileList[0], index=False, header=False, sep=delimiter
            )

        # save slice record
        fileList += [file + '_sliceRec' + ext]
        DataFrame(self.record).to_csv(
            fileList[1], index=False, header=False, sep=delimiter
            )

        # if z is complex, split into mag / phase files
        if np.iscomplexobj(self.z) and cplxSplit:
            fileList += [file + '_mag' + ext]
            fileList += [file + '_phase' + ext]
            DataFrame(np.abs(self.z)).to_csv(
                fileList[2], index=False, header=False, sep=delimiter
                )
            DataFrame(np.angle(self.z)).to_csv(
                fileList[3], index=False, header=False, sep=delimiter
                )
        else:
            fileList += [file + ext]
            DataFrame(self.z).to_csv(
                fileList[2], index=False, header=False, sep=delimiter
                )
        return fileList

    def to_csv(self, filename='', cplxSplit=True):
        return self.to_file(filename, '.csv', ',', cplxSplit)

    def to_dat(self, filename='', cplxSplit=True):
        return self.to_file(filename, '.dat', '\t', cplxSplit)


if __name__ == '__main__':
    # x, y, z, xname, yname, zname, record = get_Digitizer_data()
    # x, y, z, xname, yname, zname, record = get_VNA_Data()
    # x, yaxisList, z, ynamelist, zname, yunitlist, zunit = open_hdf5()
    # a = hdf5Handle('Frequency', 'Hz')
    # a = hdf5Handle('Time', 's')
    # a.slice()
    # a.to_csv('try.csv')
    # a.to_dat('try.dat')
    # x, y, z, xname, yname, zname, record = a.output()
    # plot(x, y, z, xname, yname, zname)
    pass
