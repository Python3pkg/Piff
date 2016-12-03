# Copyright (c) 2016 by Mike Jarvis and the other collaborators on GitHub at
# https://github.com/rmjarvis/Piff  All rights reserved.
#
# Piff is free software: Redistribution and use in source and binary forms
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

"""
.. module:: twod_stats

"""

from __future__ import print_function
import numpy as np

from .stats import Stats

class TwoDHistStats(Stats):
    """Statistics class that can make pretty colormaps where each bin has some
    arbitrary function applied to it.

    By default this will make a color map based on u and v coordinates of the
    input stars. The color scale is based on (by default) the median value of
    the objects in particular u-v voxel.

    After a call to :func:`compute`, the following attributes are accessible:

        :twodhists:     A dictionary of two dimensional histograms, with keys
                        ['u', 'v',
                         'T', 'g1', 'g2',
                         'T_model', 'g1_model', 'g2_model',
                         'dT', 'dg1', 'dg2']

    These histograms are two dimensional masked arrays where the value of the
    pixel corresponds to reducing_function([objects in u-v voxel])
    """

    def __init__(self, number_bins_u=11, number_bins_v=22, reducing_function='np.median', file_name=None, logger=None):
        """
        :param number_bins_u:       Number of bins in u direction [default: 11]
        :param number_bins_v:       Number of bins in v direction [default: 22]
        :param reducing_function:   Type of function to apply to grouped objects. numpy functions are prefixed by np. [default: 'np.median']
        :param file_name:   Name of the file to output to. [default: None]
        :param logger:      A logger object for logging debug info. [default: None]
        """
        self.number_bins_u = number_bins_u
        self.number_bins_v = number_bins_v
        self.reducing_function = eval(reducing_function)

        self.file_name = file_name

    def compute(self, psf, stars, logger=None):
        """
        :param psf:         A PSF Object
        :param stars:       A list of Star instances.
        :param logger:      A logger object for logging debug info. [default: None]
        """

        # get the shapes
        if logger:
            logger.info("Obtaining Star Model Parameters")

        # Pull out the positions
        positions = np.array([ (star.data.properties['u'], star.data.properties['v'])
                               for star in stars ])
        u, v = positions.T

        # obtain params of psf fit
        params_star = np.array([star.fit.params for star in stars])
        params_model = np.array([psf.drawStar(star).fit.params for star in stars])

        # compute the indices
        if logger:
            logger.info("Computing TwoDHist indices")

        # fudge the bins by multiplying 1.01 so that the max entries are in the bins
        self.bins_u = np.linspace(np.min(u), np.max(u) * 1.01, num=self.number_bins_u)
        self.bins_v = np.linspace(np.min(v), np.max(v) * 1.01, num=self.number_bins_v)

        # digitize u and v. No such thing as entries below their min, so -1 to index
        indx_u = np.digitize(u, self.bins_u) - 1
        indx_v = np.digitize(v, self.bins_v) - 1

        # get unique indices
        unique_indx = np.vstack({tuple(row) for row in np.vstack((indx_u, indx_v)).T})

        # compute the arrays
        if logger:
            logger.info("Computing TwoDHist arrays")
        self.twodhists = {}

        for indx in range(len(params_star.T)):
            p = self._array_to_2dhist(params_star[:, indx], indx_u, indx_v, unique_indx)
            p_model = self._array_to_2dhist(params_model[:, indx], indx_u, indx_v, unique_indx)
            dp = self._array_to_2dhist(params_star[:, indx] - params_model[:, indx], indx_u, indx_v, unique_indx)
            self.twodhists['p{0}'.format(indx)] = [p, p_model, dp]

    def plot(self, logger=None, **kwargs):
        """Make the plots.

        :param logger:      A logger object for logging debug info. [default: None]
        :params **kwargs:   Any additional kwargs go into the matplotlib plot() function.
                            [ignored in this function]

        :returns: fig, ax
        """
        import matplotlib.pyplot as plt

        ncols = len(self.twodhists)
        nrows = 3
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(4 * nrows, 3 * ncols))
        # make the colormaps
        if logger:
            logger.info("Creating TwoDHist colormaps")
        for iy in range(ncols):

            pkey = 'p{0}'.format(iy)
            vmin = np.ma.min([self.twodhists[pkey][:2]])
            vmin_d = np.ma.min([self.twodhists[pkey][2]])
            vmax = np.ma.max([self.twodhists[pkey][:2]])
            vmax_d = np.ma.max([self.twodhists[pkey][2]])
            vmid = np.ma.mean([self.twodhists[pkey][:2]])
            vmid_d = np.ma.mean([self.twodhists[pkey][2]])
            cmap = self._shift_cmap(vmin, vmax, vmid)
            cmap_d = self._shift_cmap(vmin_d, vmax_d, vmid_d)

            for ix in range(nrows):
                ax = axs[ix, iy]
                if iy == 0:
                    ax.set_ylabel('v')
                ax.set_xlim(min(self.bins_u), max(self.bins_u))
                ax.set_ylim(min(self.bins_v), max(self.bins_v))

                # diff row
                if ix == nrows - 1:
                    ax.set_xlabel('u')
                    IM = ax.pcolor(self.bins_u, self.bins_v, self.twodhists[pkey][ix], cmap=cmap_d, vmin=vmin_d, vmax=vmax_d)
                else:
                    IM = ax.pcolor(self.bins_u, self.bins_v, self.twodhists[pkey][ix], cmap=cmap, vmin=vmin, vmax=vmax)

                fig.colorbar(IM, ax=ax)

        plt.tight_layout()

        return fig, axs

    def _array_to_2dhist(self, z, indx_u, indx_v, unique_indx):
        C = np.ma.zeros((self.number_bins_v - 1, self.number_bins_u - 1))
        C.mask = np.ones((self.number_bins_v - 1, self.number_bins_u - 1))

        for unique in unique_indx:
            ui, vi = unique

            sample = z[(indx_u == ui) & (indx_v == vi)]
            if len(sample) > 0:
                value = self.reducing_function(sample)
                C[vi, ui] = value
                C.mask[vi, ui] = 0

        return C

    def _shift_cmap(self, vmin, vmax, vmid=0):
        import matplotlib.pyplot as plt
        midpoint = (vmid - vmin) / (vmax - vmin)

        # if b <= 0, then we want Blues_r
        if vmax <= 0 and vmid == 0:
            return plt.cm.Blues_r
        # if a >= 0, then we want Reds
        elif vmin >= 0 and vmid == 0:
            return plt.cm.Reds
        else:
            return self._shiftedColorMap(plt.cm.RdBu_r, midpoint=midpoint)

    def _shiftedColorMap(self, cmap, start=0, midpoint=0.5, stop=1.0,
                         name='shiftedcmap'):
        '''
        Taken from

        https://github.com/olgabot/prettyplotlib/blob/master/prettyplotlib/colors.py

        which makes beautiful plots by the way


        Function to offset the "center" of a colormap. Useful for
        data with a negative min and positive max and you want the
        middle of the colormap's dynamic range to be at zero

        Input
        -----
          cmap : The matplotlib colormap to be altered
          start : Offset from lowest point in the colormap's range.
              Defaults to 0.0 (no lower ofset). Should be between
              0.0 and `midpoint`.
          midpoint : The new center of the colormap. Defaults to
              0.5 (no shift). Should be between 0.0 and 1.0. In
              general, this should be  1 - vmax/(vmax + abs(vmin))
              For example if your data range from -15.0 to +5.0 and
              you want the center of the colormap at 0.0, `midpoint`
              should be set to  1 - 5/(5 + 15)) or 0.75
          stop : Offset from highets point in the colormap's range.
              Defaults to 1.0 (no upper ofset). Should be between
              `midpoint` and 1.0.
        '''
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        cdict = {
            'red': [],
            'green': [],
            'blue': [],
            'alpha': []
        }

        # regular index to compute the colors
        reg_index = np.linspace(start, stop, 257)

        # shifted index to match the data
        shift_index = np.hstack([
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True)
        ])

        for ri, si in zip(reg_index, shift_index):
            r, g, b, a = cmap(ri)

            cdict['red'].append((si, r, r))
            cdict['green'].append((si, g, g))
            cdict['blue'].append((si, b, b))
            cdict['alpha'].append((si, a, a))

        newcmap = LinearSegmentedColormap(name, cdict)

        # add some overunders
        newcmap.set_bad(color='g', alpha=0.75)
        newcmap.set_over(color='m', alpha=0.75)
        newcmap.set_under(color='c', alpha=0.75)

        plt.register_cmap(cmap=newcmap)

        return newcmap

class WhiskerStats(Stats):
    """Statistics class that can make whiskerplots.

    By default this will make a whisker plot based on u and v coordinates of
    the input stars. The whisker scale is based on (by default) the median
    value of the objects in a particular u-v voxel.

    After a call to :func:`compute`, the following attributes are accessible:

        :twodhists:     A dictionary of two dimensional histograms, with keys
                        ['u', 'v',
                         'w1', 'w2',
                         'w1_model', 'w2_model',
                         'dw1', 'dw2']

    These histograms are two dimensional masked arrays where the value of the
    pixel corresponds to reducing_function([objects in u-v voxel])

    Note: There are a couple different ways to define your whiskers. Here we
    have taken the approach that the whisker represents the ellipticity as:

        theta = arctan(e2, e1) / 2
        r = sqrt(e1 ** 2 + e2 ** 2)
        w1 = r cos(theta)
        w2 = r sin(theta)

    Because e1, e2 do not have units, w does not either.
    """
    def __init__(self, number_bins_u=11, number_bins_v=22, reducing_function='np.median', file_name=None, logger=None):
        """
        :param number_bins_u:       Number of bins in u direction [default: 11]
        :param number_bins_v:       Number of bins in v direction [default: 22]
        :param reducing_function:   Type of function to apply to grouped objects. numpy functions are prefixed by np. [default: 'np.median']
        :param file_name:   Name of the file to output to. [default: None]
        :param logger:      A logger object for logging debug info. [default: None]
        """
        self.number_bins_u = number_bins_u
        self.number_bins_v = number_bins_v
        self.reducing_function = eval(reducing_function)

        self.file_name = file_name

    def compute(self, psf, stars, logger=None):
        """
        :param psf:         A PSF Object
        :param stars:       A list of Star instances.
        :param logger:      A logger object for logging debug info. [default: None]
        """

        # get the shapes
        if logger:
            logger.info("Measuring Star and Model Shapes")
        positions, shapes_truth, shapes_model = self.measureShapes(psf, stars, logger=logger)

        # Only use stars for which hsm was successful
        flag_truth = shapes_truth[:, 6]
        flag_model = shapes_model[:, 6]
        mask = (flag_truth == 0) & (flag_model == 0)

        # define terms for the catalogs
        u = positions[mask, 0]
        v = positions[mask, 1]
        T = shapes_truth[mask, 3]
        g1 = shapes_truth[mask, 4]
        g2 = shapes_truth[mask, 5]
        T_model = shapes_model[mask, 3]
        g1_model = shapes_model[mask, 4]
        g2_model = shapes_model[mask, 5]
        dT = T - T_model
        dg1 = g1 - g1_model
        dg2 = g2 - g2_model

        mag_w = np.sqrt(np.square(g1) + np.square(g2))
        phi = np.arctan2(g2, g1) / 2.
        w1 = mag_w * np.cos(phi)
        w2 = mag_w * np.sin(phi)
        mag_w_model = np.sqrt(np.square(g1_model) + np.square(g2_model))
        phi_model = np.arctan2(g2_model, g1_model) / 2.
        w1_model = mag_w_model * np.cos(phi_model)
        w2_model = mag_w_model * np.sin(phi_model)
        dmag_w = np.sqrt(np.square(dg1) + np.square(dg2))
        dphi = np.arctan2(dg2, dg1) / 2.
        dw1 = dmag_w * np.cos(dphi)
        dw2 = dmag_w * np.sin(dphi)

        # compute the indices
        if logger:
            logger.info("Computing TwoDHist indices")

        # fudge the bins by multiplying 1.01 so that the max entries are in the bins
        self.bins_u = np.linspace(np.min(u), np.max(u) * 1.01, num=self.number_bins_u)
        self.bins_v = np.linspace(np.min(v), np.max(v) * 1.01, num=self.number_bins_v)

        # digitize u and v. No such thing as entries below their min, so -1 to index
        indx_u = np.digitize(u, self.bins_u) - 1
        indx_v = np.digitize(v, self.bins_v) - 1

        # get unique indices
        unique_indx = np.vstack({tuple(row) for row in np.vstack((indx_u, indx_v)).T})

        # compute the arrays
        if logger:
            logger.info("Computing TwoDHist arrays")
        self.twodhists = {}

        self.twodhists['u'] = self._array_to_2dhist(u, indx_u, indx_v, unique_indx)
        self.twodhists['v'] = self._array_to_2dhist(v, indx_u, indx_v, unique_indx)

        # w1
        self.twodhists['w1'] = self._array_to_2dhist(w1, indx_u, indx_v, unique_indx)

        # w2
        self.twodhists['w2'] = self._array_to_2dhist(w2, indx_u, indx_v, unique_indx)

        # w1_model
        self.twodhists['w1_model'] = self._array_to_2dhist(w1_model, indx_u, indx_v, unique_indx)

        # w2_model
        self.twodhists['w2_model'] = self._array_to_2dhist(w2_model, indx_u, indx_v, unique_indx)

        # dw1
        self.twodhists['dw1'] = self._array_to_2dhist(dw1, indx_u, indx_v, unique_indx)

        # dw2
        self.twodhists['dw2'] = self._array_to_2dhist(dw2, indx_u, indx_v, unique_indx)

    def plot(self, logger=None, **kwargs):
        """Make the plots.

        :param logger:      A logger object for logging debug info. [default: None]
        :params **kwargs:   Any additional kwargs go into the matplotlib plot() function.
                            [ignored in this function]

        :returns: fig, ax
        """
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, sharey=True, subplot_kw={'aspect' : 'equal'})
        axs[0].set_xlabel('u')
        axs[0].set_ylabel('v')
        axs[1].set_xlabel('u')
        axs[1].set_ylabel('v')

        # make the plots
        if logger:
            logger.info("Creating TwoDHist whiskerplots")

        # configure to taste
        # bigger scale = smaller whiskers
        quiver_dict = dict(alpha=1,
                           angles='uv',
                           headlength=0,
                           headwidth=0,
                           headaxislength=0,
                           minlength=0,
                           pivot='middle',
                           scale_units='xy',
                           width=0.001,
                           color='blue',
                           )

        # raw whiskers
        ax = axs[0]
        # data
        Q = ax.quiver(self.twodhists['u'], self.twodhists['v'], self.twodhists['w1'], self.twodhists['w2'], scale=2.5e-3, **quiver_dict)
        # quiverkey
        ax.quiverkey(Q, 0.10, 0.10, 0.03, 'e = 0.03', coordinates='axes', color='darkred', labelcolor='darkred', labelpos='S')

        # residual whiskers
        ax = axs[1]

        # dw
        Q = ax.quiver(self.twodhists['u'], self.twodhists['v'], self.twodhists['dw1'], self.twodhists['dw2'], scale=4.0e-4, **quiver_dict)
        # quiverkey
        ax.quiverkey(Q, 0.90, 0.10, 0.03, 'de = 0.03', coordinates='axes', color='darkred', labelcolor='darkred', labelpos='S')

        plt.tight_layout()

        return fig, axs

    def _array_to_2dhist(self, z, indx_u, indx_v, unique_indx):
        C = np.ma.zeros((self.number_bins_v - 1, self.number_bins_u - 1))
        C.mask = np.ones((self.number_bins_v - 1, self.number_bins_u - 1))

        for unique in unique_indx:
            ui, vi = unique

            sample = z[(indx_u == ui) & (indx_v == vi)]
            if len(sample) > 0:
                value = self.reducing_function(sample)
                C[vi, ui] = value
                C.mask[vi, ui] = 0

        return C
