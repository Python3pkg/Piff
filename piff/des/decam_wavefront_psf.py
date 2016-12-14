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
.. module:: decam_wavefront_psf
"""

from __future__ import print_function

import numpy as np

import galsim

from ..star import Star, StarFit, StarData
from ..model import Model
from ..interp import Interp
from ..outliers import Outliers
from ..psf import PSF

from ..gsobject_model import Gaussian
from ..optical_model import Optical
from .decam_wavefront import DECamWavefront
from .decaminfo import DECamInfo

from .donutengine_model import DonutEngine

from time import time

class DECamWavefrontPSF(PSF):
    """A PSF class that uses the decam wavefront to model the PSF

    Then, it should take another PSF to fit the residual...

    We need to fit the following variables:
        Constant optical sigma or kolmogorov, g1, g2 in model
        misalignments in interpolant: these are the interp.misalignment terms
    """
    def __init__(self, knn_file_name, knn_extname, max_stars=0, error_estimate=0.001, pupil_plane_im=None,  extra_interp_properties=None, weights=np.array([0.5, 1, 1]), minuit_kwargs={}, interp_kwargs={}, model_kwargs={}, verbose=0, engine='galsim', template='des'):
        """


        :param max_stars:                   [default: 0] If > 0, randomly sample only max_stars for the fit
        :param error_estimate:              [default: 0.01] Fudge factor in chi square until we get real errors for the hsm algorithm
        :param extra_interp_properties:     A list of any extra properties that will be used for
                                            the interpolation in addition to (u,v).
                                            [default: None]
        :param weights:                     Array or list of weights for comparing gaussian shapes in fit
                                            [default: [0.5, 1, 1], so downweight size]
        :param minuit_kwargs:               kwargs to pass to minuit
        """

        self.interp_kwargs = {'n_neighbors': 15, 'algorithm': 'auto'}
        self.interp_kwargs.update(interp_kwargs)

        # it turns out this part can also be slow!
        self.interp = DECamWavefront(knn_file_name, knn_extname, **self.interp_kwargs)
        self.model_comparer = Gaussian(fastfit=True, force_model_center=True, include_pixel=True)

        self.decaminfo = DECamInfo()

        self.weights = np.array(weights)
        # normalize weights
        self.weights /= self.weights.sum()

        if extra_interp_properties is None:
            self.extra_interp_properties = []
        else:
            self.extra_interp_properties = extra_interp_properties

        self.kwargs = {
            'pupil_plane_im': pupil_plane_im,
            'knn_file_name': knn_file_name,
            'knn_extname': knn_extname,
            'minuit': minuit_kwargs,
            'verbose': verbose,
            'max_stars': max_stars,
            'error_estimate': error_estimate,
            }

        # load up the model after kwargs are set
        # donutengine ends up not being much faster in this framework :(
        self._engines = ['galsim', 'galsim_fast', 'galsim_faster', 'galsim_fast_double', 'galsim_faster_double']
        # ORDER CORRESPONDS TO iTelescope FOR DONUTLIB!
        self._templates = ['des', 'MosaicII', 'lsst']
        self._model(template=template, engine=engine, **model_kwargs)


        # put in the variable names and initial values
        # TODO: This should be called from function
        self.minuit_kwargs = {
            'throw_nan': False,
            'pedantic': True,
            'print_level': int(self.kwargs['verbose']),
            'errordef': 0.01,  # guesstimated

            'r0': 0.15, 'fix_r0': False,   'limit_r0': (0.08, 0.25), 'error_r0': 1e-2,
            'g1': 0,   'fix_g1': False,   'limit_g1': (-0.2, 0.2),  'error_g1': 1e-2,
            'g2': 0,   'fix_g2': False,   'limit_g2': (-0.2, 0.2),  'error_g2': 1e-2,
            'z04d': 0, 'fix_z04d': False, 'limit_z04d': (-2, 2),
            'error_z04d': 1e-2,
            'z04x': 0, 'fix_z04x': False, 'limit_z04x': (-2, 2),
            'error_z04x': 1e-4,
            'z04y': 0, 'fix_z04y': False, 'limit_z04y': (-2, 2),
            'error_z04x': 1e-4,
            'z05d': 0, 'fix_z05d': False, 'limit_z05d': (-2, 2),
            'error_z05d': 1e-2,
            'z05x': 0, 'fix_z05x': False, 'limit_z05x': (-2, 2),
            'error_z05x': 1e-4,
            'z05y': 0, 'fix_z05y': False, 'limit_z05y': (-2, 2),
            'error_z05x': 1e-4,
            'z06d': 0, 'fix_z06d': False, 'limit_z06d': (-2, 2),
            'error_z06d': 1e-2,
            'z06x': 0, 'fix_z06x': False, 'limit_z06x': (-2, 2),
            'error_z06x': 1e-4,
            'z06y': 0, 'fix_z06y': False, 'limit_z06y': (-2, 2),
            'error_z06x': 1e-4,
            'z07d': 0, 'fix_z07d': False, 'limit_z07d': (-2, 2),
            'error_z07d': 1e-2,
            'z07x': 0, 'fix_z07x': False, 'limit_z07x': (-2, 2),
            'error_z07x': 1e-4,
            'z07y': 0, 'fix_z07y': False, 'limit_z07y': (-2, 2),
            'error_z07x': 1e-4,
            'z08d': 0, 'fix_z08d': False, 'limit_z08d': (-2, 2),
            'error_z08d': 1e-2,
            'z08x': 0, 'fix_z08x': False, 'limit_z08x': (-2, 2),
            'error_z08x': 1e-4,
            'z08y': 0, 'fix_z08y': False, 'limit_z08y': (-2, 2),
            'error_z08x': 1e-4,
            'z09d': 0, 'fix_z09d': False, 'limit_z09d': (-2, 2),
            'error_z09d': 1e-2,
            'z09x': 0, 'fix_z09x': False, 'limit_z09x': (-2, 2),
            'error_z09x': 1e-4,
            'z09y': 0, 'fix_z09y': False, 'limit_z09y': (-2, 2),
            'error_z09x': 1e-4,
            'z10d': 0, 'fix_z10d': False, 'limit_z10d': (-2, 2),
            'error_z10d': 1e-2,
            'z10x': 0, 'fix_z10x': False, 'limit_z10x': (-2, 2),
            'error_z10x': 1e-4,
            'z10y': 0, 'fix_z10y': False, 'limit_z10y': (-2, 2),
            'error_z10x': 1e-4,
            'z11d': 0, 'fix_z11d': False, 'limit_z11d': (-2, 2),
            'error_z11d': 1e-2,
            'z11x': 0, 'fix_z11x': False, 'limit_z11x': (-2, 2),
            'error_z11x': 1e-4,
            'z11y': 0, 'fix_z11y': False, 'limit_z11y': (-2, 2),
            'error_z11x': 1e-4,
                              }

        self.minuit_kwargs.update(self.kwargs['minuit'])

        self.update_psf_params(**self.minuit_kwargs)

        self._time = time()

    def _model(self, template='des', engine='galsim', **model_kwargs):
        if engine == 'galsim_fast':
            # pass in gsparams object to speed up everything
            gsparams = galsim.GSParams(minimum_fft_size=32,  # 128
                                       # maximum_fft_size=4096,  # 4096
                                       # stepk_minimum_hlr=5,  # 5
                                       # folding_threshold=5e-3,  # 5e-3
                                       # maxk_threshold=1e-3,  # 1e-3
                                       # kvalue_accuracy=1e-5,  # 1e-5
                                       # xvalue_accuracy=1e-5,  # 1e-5
                                       # table_spacing=1.,  # 1
                                       )
            # padfactor?
            pad_factor = 0.5
            oversampling = 0.5
            self.model = Optical(template=template, pupil_plane_im=self.kwargs['pupil_plane_im'], gsparams=gsparams, pad_factor=pad_factor, oversampling=oversampling, **model_kwargs)
            self.model = Optical(template=template, pupil_plane_im=self.kwargs['pupil_plane_im'], gsparams=gsparams, pad_factor=pad_factor, oversampling=oversampling, **model_kwargs)
        elif engine == 'galsim_fast_double':
            # pass in gsparams object to speed up everything
            gsparams = galsim.GSParams(minimum_fft_size=32,  # 128
                                       # maximum_fft_size=1024,  # 4096
                                       # stepk_minimum_hlr=5,  # 5
                                       # folding_threshold=5e-3,  # 5e-3
                                       # maxk_threshold=1e-3,  # 1e-3
                                       # kvalue_accuracy=1e-5,  # 1e-5
                                       # xvalue_accuracy=1e-5,  # 1e-5
                                       # table_spacing=1.,  # 1
                                       )
            # padfactor?
            pad_factor = 0.5
            oversampling = 0.5
            self.model = Optical(scale_optical_lambda=2.0, template=template, pupil_plane_im=self.kwargs['pupil_plane_im'], gsparams=gsparams, pad_factor=pad_factor, oversampling=oversampling, **model_kwargs)
        elif engine == 'galsim_faster':
            # pass in gsparams object to speed up everything
            gsparams = galsim.GSParams(minimum_fft_size=32,  # 128
                                       # maximum_fft_size=4096,  # 4096
                                       # stepk_minimum_hlr=5,  # 5
                                       # folding_threshold=5e-3,  # 5e-3
                                       # maxk_threshold=1e-3,  # 1e-3
                                       # kvalue_accuracy=1e-5,  # 1e-5
                                       # xvalue_accuracy=1e-5,  # 1e-5
                                       # table_spacing=1.,  # 1
                                       )
            # padfactor?
            pad_factor = 0.25
            oversampling = 0.5
            self.model = Optical(template=template, pupil_plane_im=self.kwargs['pupil_plane_im'], gsparams=gsparams, pad_factor=pad_factor, oversampling=oversampling, **model_kwargs)
        elif engine == 'galsim_faster_double':
            # pass in gsparams object to speed up everything
            gsparams = galsim.GSParams(minimum_fft_size=32,  # 128
                                       # maximum_fft_size=512,  # 4096
                                       # stepk_minimum_hlr=5,  # 5
                                       # folding_threshold=5e-3,  # 5e-3
                                       # maxk_threshold=1e-3,  # 1e-3
                                       # kvalue_accuracy=1e-5,  # 1e-5
                                       # xvalue_accuracy=1e-5,  # 1e-5
                                       # table_spacing=1.,  # 1
                                       )
            # padfactor?
            pad_factor = 0.25
            oversampling = 0.5
            self.model = Optical(scale_optical_lambda=2.0, template=template, pupil_plane_im=self.kwargs['pupil_plane_im'], gsparams=gsparams, pad_factor=pad_factor, oversampling=oversampling, **model_kwargs)
        elif engine == 'galsim':
            self.model = Optical(template=template, pupil_plane_im=self.kwargs['pupil_plane_im'], **model_kwargs)
        elif engine == 'donutlib_fast':
            makedonut_dict = {'nbin': 192,  # 256
                              'nPixels': 24,  # 32
                              'pixelOverSample': 8,  # 8
                              'scaleFactor': 1,  # 1
                              'randomFlag': False,
                              'iTelescope': self._templates.index(template),
                              'nZernikeTerms': 11,
                              }
            self.model = DonutEngine(**makedonut_dict)
        elif engine == 'donutlib_fast_scalefactor':
            makedonut_dict = {'nbin': 96,  # 256
                              'nPixels': 24,  # 32
                              'pixelOverSample': 4,  # 8
                              'scaleFactor': 2,  # 1
                              'randomFlag': False,
                              'iTelescope': self._templates.index(template),
                              'nZernikeTerms': 11,
                              }
            self.model = DonutEngine(**makedonut_dict)
        elif engine == 'donutlib':
            makedonut_dict = {'nbin': 384,  # 256
                              'nPixels': 24,  # 32
                              'pixelOverSample': 16,  # 8
                              'scaleFactor': 1,  # 1
                              'randomFlag': False,
                              'iTelescope': self._templates.index(template),
                              'nZernikeTerms': 11,
                              }
            self.model = DonutEngine(**makedonut_dict)
        elif engine == 'donutlib_again':
            makedonut_dict = {'nbin': 384,  # 256
                              'nPixels': 24,  # 32
                              'pixelOverSample': 16,  # 8
                              'scaleFactor': 1,  # 1
                              'randomFlag': False,
                              'iTelescope': self._templates.index(template),
                              'nZernikeTerms': 11,
                              }
            self.model = DonutEngine(**makedonut_dict)
        elif engine == 'donutlib_scalefactor':
            makedonut_dict = {'nbin': 384,  # 256
                              'nPixels': 24,  # 32
                              'pixelOverSample': 16,  # 8
                              'scaleFactor': 2,  # 1
                              'randomFlag': False,
                              'iTelescope': self._templates.index(template),
                              'nZernikeTerms': 11,
                              }
            self.model = DonutEngine(**makedonut_dict)
        elif engine == 'donutlib_old':
            makedonut_dict = {'nbin': 256,  # 256
                              'nPixels': 32,  # 32
                              'pixelOverSample': 8,  # 8
                              'scaleFactor': 1,  # 1
                              'randomFlag': False,
                              'iTelescope': self._templates.index(template),
                              'nZernikeTerms': 11,
                              }
            self.model = DonutEngine(**makedonut_dict)
        else:
            raise Exception('Invalid engine! {0}'.format(engine))

    def _measure_shapes(self, stars, logger=None, reject=False):
        """Work around the gsobject to measure shapes. Returns stars
        """
        stars_out = []
        # TODO: It would be nice if I could copy the stars so that the new list is not the same as the old...
        for star_i, star in enumerate(stars):
            if logger:
                logger.info("Measuring shape of star {0}".format(star_i))
                logger.info(star.data.properties)
            star.fit.params = None
            try:
                star_out = self.model_comparer.fit(star, logger=logger)
                if logger:
                    logger.info(star_out.fit.params)
                stars_out.append(star_out)
            except:
                if logger:
                    logger.info("Star rejected!")
                if not reject:
                    # not supposed to be rejecting stars...
                    star_out = self.model_comparer.fit(star, logger=logger)
                    if logger:
                        logger.info(star_out.fit.params)
                    stars_out.append(star_out)
        return stars_out

    def fit(self, stars, wcs, pointing,
            chisq_threshold=0.1, max_iterations=300, skip_fit=False, logger=None):
        """Fit interpolated PSF model to star data using standard sequence of operations.

        :param stars:           A list of Star instances.
        :param wcs:             A dict of WCS solutions indexed by chipnum.
        :param pointing:        A galsim.CelestialCoord object giving the telescope pointing.
                                [Note: pointing should be None if the WCS is not a CelestialWCS]
        :param chisq_threshold: Change in reduced chisq at which iteration will terminate.
                                [default: 0.1]
        :param max_iterations:  Maximum number of iterations to try. [default: 300]
        :param skip_fit:        If True, do not run migrad fit [default: False]
        :param logger:          A logger object for logging debug info. [default: None]
        """
        if logger:
            logger.info("Start fitting DECAMWavefrontPSF using %s stars", len(stars))
        from iminuit import Minuit

        self.stars = stars
        if self.kwargs['max_stars'] and self.kwargs['max_stars'] < len(self.stars):
            choice = np.random.choice(len(self.stars), self.kwargs['max_stars'])
            self.stars = [stars[i] for i in choice]
        self.wcs = wcs
        self.pointing = pointing

        self._n_iter = 0

        # get the moments of the stars for comparison
        if logger:
            logger.info("Start measuring the shapes")
        self._stars = self._measure_shapes(self.stars, logger=logger, reject=True)
        self._shapes = np.array([star.fit.params for star in self._stars])
        # convert shapes to unnormalized second moments
        self._shapes[:,1:] = self._shapes[:,0][:, None] * self._shapes[:,1:]

        if logger:
            logger.info("Creating minuit object")
            self._logger = logger
        else:
            self._logger = None
        self._minuit = Minuit(self._fit_func, **self.minuit_kwargs)
        # run the fit and solve! This will update the interior parameters
        if logger:
            logger.info("Running migrad!")
        self._minuit.migrad(ncall=max_iterations)
        # self._hesse = self._minuit.hesse()
        # self._minos = self._minuit.minos()
        # these are the best fit parameters
        self._fitarg = self._minuit.fitarg
        # update params to best values
        if logger:
            logger.info("Fitargs are {0}".format(self._fitarg))
            logger.info("Minuit values are {0}".format(self._minuit.values))
        self.update_psf_params(**self._minuit.values)
        # save params and errors to the kwargs
        self.kwargs['minuit'].update(self._fitarg)
        if logger:
            logger.info("Minuit kwargs are now: {0}".format(self.kwargs['minuit']))

    def update_psf_params(self,
                          r0=np.nan, g1=np.nan, g2=np.nan,
                          z04d=np.nan, z04x=np.nan, z04y=np.nan,
                          z05d=np.nan, z05x=np.nan, z05y=np.nan,
                          z06d=np.nan, z06x=np.nan, z06y=np.nan,
                          z07d=np.nan, z07x=np.nan, z07y=np.nan,
                          z08d=np.nan, z08x=np.nan, z08y=np.nan,
                          z09d=np.nan, z09x=np.nan, z09y=np.nan,
                          z10d=np.nan, z10x=np.nan, z10y=np.nan,
                          z11d=np.nan, z11x=np.nan, z11y=np.nan,
                          logger=None, **kwargs):
        # update model
        if r0 == r0:
            self.model.kolmogorov_kwargs['r0'] = r0
        if g1 == g1:
            self.model.g1 = g1
        if g2 == g2:
            self.model.g2 = g2
        # update the misalignment
        misalignment = np.array([
                  [z04d, z04x, z04y],
                  [z05d, z05x, z05y],
                  [z06d, z06x, z06y],
                  [z07d, z07x, z07y],
                  [z08d, z08x, z08y],
                  [z09d, z09x, z09y],
                  [z10d, z10x, z10y],
                  [z11d, z11x, z11y],
                  ])
        old_misalignment = self.interp.misalignment
        misalignment = np.where(misalignment == misalignment, misalignment, old_misalignment)
        self.interp.misalignment = misalignment

        if logger:
            logger.info('Old misalignment is {0}'.format(old_misalignment))
            logger.info('New misalignment is {0}'.format(misalignment))

    def _fit_func(self,
                  r0, g1, g2,
                  z04d, z04x, z04y,
                  z05d, z05x, z05y,
                  z06d, z06x, z06y,
                  z07d, z07x, z07y,
                  z08d, z08x, z08y,
                  z09d, z09x, z09y,
                  z10d, z10x, z10y,
                  z11d, z11x, z11y,
                  ):

        logger = self._logger
        # update psf
        self.update_psf_params(r0, g1, g2,
                               z04d, z04x, z04y,
                               z05d, z05x, z05y,
                               z06d, z06x, z06y,
                               z07d, z07x, z07y,
                               z08d, z08x, z08y,
                               z09d, z09x, z09y,
                               z10d, z10x, z10y,
                               z11d, z11x, z11y,
                               logger=logger,
                               )

        # get shapes
        for ith, star in enumerate(self._stars):
            star_out = self.drawStar(star)
            # save star
        shapes = np.array([star.fit.params for star in self._measure_shapes(self.drawStarList(self._stars), logger=logger, reject=False)])
        shapes[:, 1:] = shapes[:,0][:, None] * shapes[:, 1:]

        # calculate chisq
        # TODO: are there any errors from the shape measurements I could put in?
        chi2 = np.sum(np.square((shapes - self._shapes) / self.kwargs['error_estimate']), axis=0)
        dof = shapes.size
        if self._n_iter % 10 == 0 and self.kwargs['verbose']:
            print('\n',
                    '***************************************\n',
                    'time\t {0:.3e}\t ncalls \t {1}\n'.format(time() - self._time, self._n_iter),
                    'size\t {0:.3e}\t {1:.3e}\t {2:.3e}\n'.format(r0, g1, g2),
                    'z4\t {0:.3e}\t {1:.3e}\t {2:.3e}\n'.format(z04d, z04x, z04y),
                    'z5\t {0:.3e}\t {1:.3e}\t {2:.3e}\n'.format(z05d, z05x, z05y),
                    'z6\t {0:.3e}\t {1:.3e}\t {2:.3e}\n'.format(z06d, z06x, z06y),
                    'z7\t {0:.3e}\t {1:.3e}\t {2:.3e}\n'.format(z07d, z07x, z07y),
                    'z8\t {0:.3e}\t {1:.3e}\t {2:.3e}\n'.format(z08d, z08x, z08y),
                    'z9\t {0:.3e}\t {1:.3e}\t {2:.3e}\n'.format(z09d, z09x, z09y),
                    'z10\t {0:.3e}\t {1:.3e}\t {2:.3e}\n'.format(z10d, z10x, z10y),
                    'z11\t {0:.3e}\t {1:.3e}\t {2:.3e}\n'.format(z11d, z11x, z11y),
                    'chi2\t {0:.3e}\t {1:.3e}\t {2:.3e}\n'.format(*(chi2 / dof)),
                    '***************************************',
                    )
        self._n_iter += 1
        return np.sum(self.weights * chi2) * 1. / dof / np.sum(self.weights)

    def drawStarList(self, stars):
        """Generate PSF images for given stars.

        :param stars:       List of Star instances holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.

        :returns:           List of Star instances with its image filled with rendered PSF
        """
        # put in the focal coordinates
        stars = self.decaminfo.pixel_to_focalList(stars)
        # Interpolate parameters to this position/properties:
        stars = self.interp.interpolateList(stars)
        # Render the image
        stars = [self.model.draw(star) for star in stars]

        return stars

    def drawStar(self, star):
        """Generate PSF image for a given star.

        :param star:        Star instance holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.

        :returns:           Star instance with its image filled with rendered PSF
        """
        # put in the focal coordinates
        star = self.decaminfo.pixel_to_focal(star)
        # Interpolate parameters to this position/properties:
        star = self.interp.interpolate(star)
        # Render the image
        star = self.model.draw(star)
        return star
