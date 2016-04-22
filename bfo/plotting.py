import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.ndimage as ndimage

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


class Experiment:
    def __init__(self, experiment_number, data_folder, sample=None):
        self.experiment_number = experiment_number
        self.data_folder = data_folder
        self.sample = sample

    def __repr__(self):
        short_data_folder = "/".join(self.data_folder.split("/")[-3:])
        return "Experiment('{0.experiment_number}', ...{1}, {0.sample})"\
            .format(self, short_data_folder)


class Scan:
    """Class for loading and plotting I16 scan datafile."""

    def __init__(self, scan_number, experiment):
        self.scan_number = scan_number
        self.experiment = experiment
        self.filepath = "{0}/{1}.dat".format(
            experiment.data_folder, scan_number)
        self.metadata = self._import_metadata()
        self.data = self._import_data()

    def __repr__(self):
        return "{0.__class__.__name__}({0.scan_number}, {0.experiment})".format(self)

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
        ncols = int(np.ceil(no_scans / nrows))
        print(nrows, ncols)
        ymax = max(max(scan.data[yval]) for scan in self.scans)
        for i, scan in enumerate(self.scans):
            x, y = scan.data[xval], scan.data[yval]
            plt.subplot(nrows, ncols, i + 1)
            plt.ylim([0, ymax])
            plt.plot(x, y, **plot_kwargs)
        plt.show()

    def plot(self, xval, yval, normalisation='ic1monitor', **plot_kwargs):
        fig, ax = plt.subplots()
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

    def plot(self, xval, yval, **plot_kwargs):
        fig, ax = plt.subplots()
        ax.plot(self.scans[0].data[xval], self.average(yval), **plot_kwargs)
        ax.set_xlabel(xval)
        ax.set_ylabel(yval)
        plt.show()


class ReciprocalSpaceMap(MultiScan):
    def interpolate(self, h_offset=0, k_offset=0, l_offset=0,
                    h_point_density=200, k_point_density=200,
                    smoothing_factor=None, rotation_angle=None):

        # extract hkl values and APD data
        hkl = self.hkl_values(h_offset, k_offset, l_offset)
        if rotation_angle is None:
            h, k, l = hkl.T
        else:
            h, k, l = rotate(hkl.T, rotation_angle)
        y = np.array([yi for scan in self.scans for yi in scan.data['APD']])

        # create interpolated hkl map
        hs = np.linspace(h.min(), h.max(), h_point_density)
        ks = np.linspace(k.min(), k.max(), k_point_density)
        h_grid, k_grid = np.meshgrid(hs, ks)
        rs_map = mlab.griddata(h, k, y, hs, ks)
        # add artificial noise to fill in blanks in grid
        fake_noise = np.random.randint(0, 10, size=rs_map.shape)
        rs_map[rs_map.mask] = fake_noise[rs_map.mask]
        if smoothing_factor is not None:
            rs_map = ndimage.gaussian_filter(rs_map,
                                             sigma=smoothing_factor,
                                             order=0)

        return h_grid, k_grid, rs_map

    def plot(self, h_offset=0, k_offset=0, l_offset=0, arrows=False,
             vmin=0, vmax=40, axes=True, xlims=[-0.02, 0.02],
             ylims=[-0.02, 0.02], add_colorbar=True, save=False,
             title=False, h_point_density=200, k_point_density=200,
             color_map=plt.cm.viridis, smoothing_factor=None):

        h_grid, k_grid, rs_map = self.interpolate(h_offset, k_offset, l_offset,
                                                  h_point_density, k_point_density,
                                                  smoothing_factor)
        # plot map with eta scans superimposed
        fig, ax = plt.subplots()

        a = ax.pcolor(
            h_grid, k_grid, rs_map, vmin=vmin, vmax=vmax, cmap=color_map)
        if add_colorbar:
            plt.colorbar(a).ax.tick_params(direction='out')
        ax.scatter(0, 0, s=10)

        # adjust axes
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_xlabel('h')
        ax.set_ylabel('k')
        ax.get_xaxis().set_tick_params(direction='out')
        ax.get_yaxis().set_tick_params(direction='out')
        ax.set_title('First scan: {0}'.format(self.scan_range[0]))
        if not axes:
            plt.axis('off')

        # adjust figure dimensions
        xdim = (xlims[1] - xlims[0]) * 200
        ydim = (ylims[1] - ylims[0]) * 200
        fig.set_size_inches(xdim, ydim)
        plt.show()

    def hkl_values(self, h_offset, k_offset, l_offset):
        """Extract hkl values of all scans and merge into 1D array of
        (h, k, l) vectors expressed in the cartesian basis."""
        hkl = []
        for scan in self.scans:
            hkl_list = (scan.data['h'] - h_offset,
                        scan.data['k'] - k_offset,
                        scan.data['l'] - l_offset)
            for hi, ki, li in zip(*hkl_list):
                hkl.append(hex2cart(np.array([hi, ki, li])))
        return np.array(hkl)


# Utility functions


def rotate(vector, theta):
    """Rotate Cartesian vector by angle theta in the xy plane"""
    theta = np.radians(theta)
    rotation_matrix = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    return rotation_matrix.dot(vector)


def hex2cart(point):
    """Convert hexagonal point into Cartesian basis"""
    h, k, l = point
    return h * np.cos(np.pi / 6), k + h * np.sin(np.pi / 6), l


def xylims(x, y):
    xlims = x.min(), x.max()
    ylims = y.min(), y.max()
    return xlims, ylims
