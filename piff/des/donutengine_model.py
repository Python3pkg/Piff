
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
#    this list of conditions and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.

"""
.. module:: donutengine_model
"""

from __future__ import print_function

import galsim
import numpy as np

from ..model import Model
from ..star import Star, StarFit, StarData

class DonutEngine(Model):
    def __init__(self, logger=None, **kwargs):
        """Initialize the DonutEngine Model

        There are potentially two components to this model that are convolved together.

        First, there may be an atmospheric component, which uses a galsim.Kolmogorov to
        model the profile.

        :param r0:              The Fried parameter in units of meters to use to calculate fwhm
                                as fwhm = 0.976 lam / r0. [default: None]

        Finally, there is allowed to be a final Gaussian component and an applied shear.

        :param g1, g2:          Shear to apply to final image. Simulates vibrational modes.
                                [default: 0]

        """
        self.kwargs = {}
        self.kwargs.update(kwargs)

        self.kolmogorov_kwargs = {}
        # Store the shear parts
        self.kolmogorov_kwargs['r0'] = kwargs.pop('r0',None)
        self.g1 = kwargs.pop('g1',None)
        self.g2 = kwargs.pop('g2',None)



        # # Check that no unexpected parameters were passed in:
        # extra_kwargs = [k for k in kwargs if k not in optical_psf_keys and k not in kolmogorov_keys]
        # if len(extra_kwargs) > 0:
        #     raise TypeError('__init__() got an unexpected keyword argument %r'%extra_kwargs[0])

        self.make_makedonut(**self.kwargs)

    def make_makedonut(self, **kwargs):
        # from donutlib.makedonut import makedonut
        makedonut = None

        makedonut_dict = {'nbin': 96,  # 256
                          'nPixels': 24,  # 32
                          'pixelOverSample': 4,  # 8
                          'scaleFactor': 2,  # 1
                          'randomFlag': False,
                          'iTelescope': 0,
                          'nZernikeTerms': 11,
                          }
        # makedonut_dict = {'nbin': 384,  # 256
        #                   'nPixels': 24,  # 32
        #                   'pixelOverSample': 16,  # 8
        #                   'scaleFactor': 2,  # 1
        #                   'randomFlag': False,
        #                   }
        # update with kwargs
        makedonut_dict.update(kwargs)
        self._makedonut = makedonut(**makedonut_dict)

    def fit(self, star):
        """Warning: This method just updates the fit with the chisq and dof!

        :param star:    A Star instance

        :returns: a new Star with the fitted parameters in star.fit
        """
        image = star.image
        weight = star.weight
        # make image from self.draw
        model_image = self.draw(star).image

        # compute chisq
        chisq = np.std(image.array - model_image.array)
        dof = np.count_nonzero(weight.array) - 6

        fit = StarFit(star.fit.params, flux=star.fit.flux, center=star.fit.center, chisq=chisq, dof=dof)
        return Star(star.data, fit)

    def draw(self, star):
        """Draw the model on the given image.

        :param star:    A Star instance with the fitted parameters to use for drawing and a
                        data field that acts as a template image for the drawn model.

        :returns: a new Star instance with the data field having an image of the drawn model.

        Use donutengine to draw the image, turn into galsim profile for shearing
        """
        # create the stamp via makedonut
        # TODO: for now set x and y to 0, 0
        xDECam = star.data['focal_x']
        yDECam = star.data['focal_y']
        xDECam = 0
        yDECam = 0
        from time import time
        time0 = time()

        # old donutlib
        # stamp = self._makedonut.make(inputZernikeArray=[0] * 3 + list(star.fit.params),
        # new donutlib
        stamp = self._makedonut.make(ZernikeArray=[0] * 3 + list(star.fit.params),
                                     rzero=self.kolmogorov_kwargs['r0'],
                                     nEle=1,
                                     background=0,
                                     xDECam=xDECam,
                                     yDECam=yDECam).astype(np.float64)
        time1 = time()

        # turn the stamp into a galsim image
        # center = galsim.PositionD(*star.fit.center)
        # offset = star.data.image_pos + center - star.data.image.trueCenter()
        image = galsim.Image(stamp)
        image.wcs = star.data.image.wcs
        image.scale = star.data.image.scale
        # image.setCenter(center)
        # image.setOrigin(offset)
        # image = star.data.image.copy()
        # # is it possible to shear an image?
        # if self.g1 is not None or self.g2 is not None:
        #     image = image.shear(g1=self.g1, g2=self.g2)

        # problems:
        # 1. Need to put in the offset
        # 2. Need to put in the shear

        time2 = time()

        # TODO: might need to update image pos?
        properties = star.data.properties.copy()
        for key in ['x', 'y', 'u', 'v']:
            # Get rid of keys that constructor doesn't want to see:
            properties.pop(key,None)
        data = StarData(image=image,
                        image_pos=star.data.image_pos,
                        weight=None,
                        pointing=star.data.pointing,
                        field_pos=star.data.field_pos,
                        values_are_sb=star.data.values_are_sb,
                        properties=properties)

        time3 = time()
        star = Star(data, star.fit)
        time4 = time()
        # print('{0:.2e} {1:.2e} {2:.2e} {3:.2e}'.format(*[time1 - time0, time2 - time1, time3 - time2, time4 - time3]))
        return star


