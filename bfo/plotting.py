from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib import style
style.use('physics')

color_dict = {
    'dark_red': '#67001f', 'light_red': '#d6604d',
    'dark_blue': '#053061', 'light_blue': '#2166ac',
    'dark_green': '#006837', 'light_green': '#1a9850',
    'dark_purple': '#40004b', 'light_purple': '#762a83',
    'dark_pink': '#8e0152', 'light_pink': '#c51b7d',
    'dark_teal': '#003c30', 'light_teal': '#01665e',
    'dark_orange': '#993404', 'light_orange': '#ec7014',
    'dark_grey': '#dFdF4d', 'light_grey': '#878787',
    'dark_brown': '#543005', 'light_brown': '#8c510a',
    'dark_black': '#111111', 'light_black': '#454545',
}

Experiment = namedtuple('Experiment', 'experiment_number data_folder sample')


class Scan:
    """Class for loading and plotting I16 scan datafile."""

    def __init__(self, scan_number, experiment):
        self.scan_number = scan_number
        self.experiment = experiment
        self.filepath = "{0}/{1}.dat".format(
            experiment.data_folder, scan_number)
        self.metadata = self._import_metadata()
        self.data = self._import_data()

    def _metadata_rows(self):
        with open(self.filepath, 'r') as data_file:
            for i, line in enumerate(data_file):
                if line.find('&END') != -1:
                    return i + 1
            else:
                raise ValueError("Invalid data file {0}".format(self.filepath))

    def _import_metadata(self):
        with open(self.filepath, 'r') as data_file:
            lines = data_file.readlines()
            for i, line in enumerate(lines):
                if line.find('<MetaDataAtStart>') != -1:
                    metadata_start = i + 1
                    break
            else:
                raise ValueError("Invalid metadata in {0}".format(self.filepath))
            metadata_end = self._metadata_rows() - 2

        metadata = {}
        for line in lines[metadata_start:metadata_end]:
            field, value = line.strip('"').strip('\n').split('=')
            try:
                metadata[field] = eval(value)
            except NameError:
                metadata[field] = str(value)
        return metadata

    def _import_data(self):
        data = np.genfromtxt(fname=self.filepath,
                             skip_header=self._metadata_rows(),
                             names=True)
        return data

    def plot(self, xval, yval, normalisation='ic1monitor', **plot_kwargs):
        x, y = self.data[xval], self.data[yval]
        if normalisation:
            y *= self.data[normalisation].max() / self.data[normalisation]
        fig, ax = plt.subplots()
        ax.plot(x, y, **plot_kwargs)
        ax.set_xlabel(xval)
        ax.set_ylabel(yval)
        ax.set_title('Scan %d' % self.scan_number)
        return fig, ax


class MultiScan:
    """Class for loading and plotting multiple I16 scan datafiles"""

    def __init__(self, scan_range, experiment, scan_type=Scan):
        self.scan_range = scan_range
        self.scans = [scan_type(scan_number, experiment)
                      for scan_number in scan_range]
        self.experiment = experiment


    def grid_plot(self, xval, yval, normalisation='ic1monitor', **plot_kwargs):
        no_scans = len(self.scans)
        nrows = int(np.floor(np.sqrt(no_scans)))
        ncols = int(np.ceil(np.sqrt(no_scans)))
        fig = plt.figure()
        for i, scan in enumerate(self.scans):
            x, y = scan.data[xval], scan.data[yval]
            plt.subplot(nrows, ncols, i + 1)
            plt.plot(x, y, **plot_kwargs)
        plt.show()

    def plot(self, xval, yval, fig=None, normalisation='ic1monitor', **plot_kwargs):
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.get_axes()[0]
        for scan in self.scans:
            x, y = scan.data[xval], scan.data[yval]
            ax.plot(x, y, **plot_kwargs)
        ax.set_xlabel(xval)
        ax.set_ylabel(yval)
        ax.set_title('Scan {0}-{1}'.format(self.scan_range[0],
                                           self.scan_range[-1]))
        return fig, ax


class AverageScan(MultiScan):
    """Class for loading and plotting average of multiple I16 scan
    datafiles"""

    def average(self, yval):
        yval_arrays = []
        for scan in self.scans:
            yval_arrays.append(scan.data[yval])
        return np.array(yval_arrays).mean(axis=0)

    def plot(self, xval, yval, fig=None, **plot_kwargs):
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.get_axes()[0]
        ax.plot(self.scans[0].data[xval], self.average(yval), **plot_kwargs)
        ax.set_xlabel(xval)
        ax.set_ylabel(yval)
        plt.show()
