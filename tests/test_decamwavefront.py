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

from __future__ import print_function
import numpy as np
import piff
import os
import yaml
import subprocess

from piff_test_helper import get_script_name

import matplotlib
matplotlib.use('Agg')

from time import time

def test_init():
    load_decamwavefrontpsf()

def test_fit(engine='galsim_fast'):
    params_list = [
            { 'z04d': 0.5, },
            { 'r0': 0.14,
              'z05d': 0.5, },
            {
              'r0': 0.14,
              'z04d': 0.5,
              'z05d': 0.5,
              'z06x': 0.002,
              'z07d': -0.5,
              'z10d': -0.5,
              },
            ]
    for params_i, params in enumerate(params_list):
        # fit only a couple parameters
        n_samples = 100
        psf = load_decamwavefrontpsf(engine=engine)
        stars, wf, deltatime = generate_sample(params, n_samples, engine=engine)

        # speed things up by fixing the other keys
        for key in psf.minuit_kwargs:
            if 'fix' in key:
                if key.split('fix_')[-1] in params.keys():
                    psf.minuit_kwargs[key] = False
                else:
                    psf.minuit_kwargs[key] = True

        psf.fit(stars, None, None)

        # plot results of fit
        # for name, Stats in zip(['twodhist', 'whisker'], [piff.TwoDHistStats, piff.WhiskerStats]):
        for name, Stats in zip(['twodhist'], [piff.TwoDHistStats]):
            stats = Stats(file_name='wavefront_test/fit_{0}_{1}.png'.format(params_i, name),
                          number_bins_u=11, number_bins_v=22)
            stats.compute(psf, psf.stars)
            stats.write()

        # check fit
        for key in params:
            print('checking fit values for {0}:{1}'.format(params_i, key))
            rtol = 0.1
            np.testing.assert_allclose(params[key], psf.kwargs['minuit'][key], rtol=rtol)
            np.testing.assert_allclose(params[key], psf._fitarg[key], rtol=rtol)

        # check that if you put in the correct params you get a reasonable chisq of, um, basically zero
        fitval = psf._fit_func(**psf._minuit.values)
        values = {key: psf._minuit.values[key] for key in psf._minuit.values}
        values.update(params)
        inval = psf._fit_func(**values)
        assert inval <= fitval,"Somehow the fit does better than our input?! {0:.2e} vs {1:.2e}".format(inval, fitval)

def test_yaml():
    # piffify in code
    with open('wavefrontpsf.yaml') as f_in:
        config = yaml.load(f_in.read())
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(verbose=0)
    # Do the whole thing with the config parser
    piff.piffify(config, logger)
    psf_file = '{0}/{1}'.format(config['output']['dir'], config['output']['file_name'])
    psf = piff.read(psf_file)
    # TODO: Other tests

    # Test using the piffify executable
    os.remove(psf_file)
    config['verbose'] = 1
    piffify_exe = get_script_name('piffify')
    p = subprocess.Popen( [piffify_exe, 'wavefrontpsf.yaml'] )
    p.communicate()
    psf = piff.read(psf_file)
    import ipdb; ipdb.set_trace()

def generate_sample(params={}, n_samples=5000, engine='galsim_fast', seed=123456):

    np.random.seed(seed)
    chipnums = np.random.randint(1, 63, n_samples)
    icens = np.random.randint(1, 2048, n_samples)
    jcens = np.random.randint(1, 4096, n_samples)

    # create stars
    stars = []
    arcsecperpixel = 0.263 # arcsec per pixel
    decaminfo = piff.des.decaminfo.DECamInfo()
    # convert icen to u based on chipnums.
    # focal plane center: (u,v) = (0,0)
    # get focal coords
    xpos, ypos = decaminfo.getPosition_chipnum(chipnums, icens, jcens)
    # convert from mm to uv
    us = xpos / decaminfo.mmperpixel * arcsecperpixel
    vs = ypos / decaminfo.mmperpixel * arcsecperpixel
    for icen, jcen, chipnum, u, v in zip(icens, jcens, chipnums, us, vs):
        # we make the star smaller to speed things up
        star = piff.Star.makeTarget(x=icen, y=jcen, u=u, v=v, properties={'chipnum': chipnum}, stamp_size=32, scale=arcsecperpixel)
        stars.append(star)

    # get the focal positions
    stars = piff.des.DECamInfo().pixel_to_focalList(stars)

    psf = load_decamwavefrontpsf(engine=engine)
    psf.update_psf_params(**params)
    time0 = time()
    stars = psf.drawStarList(stars)
    time1 = time()

    return stars, psf, time1 - time0

def load_decamwavefrontpsf(engine='galsim_fast', do_pupil_plane_im=False):
    knn_file_name = 'wavefront_test/Science-20121120s1-v20i2.fits'
    knn_extname = 'Science-20121120s1-v20i2'
    if do_pupil_plane_im:
        # pupil plane image is not working yet? or I need to specify scale...
        pupil_plane_im = 'optics_test/DECam_pupil_128.fits'
        if '128' in pupil_plane_im:
            pupil_plane_scale = 12.823257 * 1. / 128
        elif '512' in pupil_plane_im:
            pupil_plane_scale = 12.823257 * 1. / 512
        else:
            pupil_plane_scale = 1.
        psf = piff.des.DECamWavefrontPSF(knn_file_name, knn_extname, pupil_plane_im, engine=engine, model_kwargs={'pupil_plane_scale': pupil_plane_scale}, verbose=2)
    else:
        psf = piff.des.DECamWavefrontPSF(knn_file_name, knn_extname, verbose=2, engine=engine)

    return psf

def test_optical_engines():
    # we have rough expectations for how the model should measure moments, and
    # we need to make sure that messing with gsparams doesn't mess that up.
    # then we also need to test that the relationship is the same whether we
    # use the reduced GSParams or the default GSParams

    params = {'r0': 0.2,
              'z04d': 0.5,
              'z05d': 0.5,
              'z07d': -0.5,
              'z06x': 0.002,
              'z10d': -0.5,
              }
    shapes = {}
    # TODO: choose engines and templates from decamwavefront class
    # TODO: galsim_faster has problems with ellipticity second moment
    engines = ['galsim_fast',
               # 'galsim_faster',
               'galsim',
               # 'galsim_faster_double', 'galsim_fast_double',
               ]
    templates = ['des', 'lsst']

    # repeat a couple times
    for template in templates:
        for engine in engines:
            for n_samples in [1, 1, 1, 10, 50, 100]:
                stars, wf, deltatime = generate_sample(params, n_samples, engine=engine)
                print('took {0:.2e} to drawStarList {1} stars for {2} on {3} template'.format(deltatime, n_samples, engine, template))

        n_samples = 50
        for engine in engines:
            # load engine
            wf_i = load_decamwavefrontpsf(engine=engine)
            wf_i.update_psf_params(**params)
            # draw stars
            stars_i = wf_i.drawStarList(stars)
            # get the shapes
            stars_i = wf_i._measure_shapes(stars_i)
            shapes[engine] = np.array([star.fit.params for star in stars_i])
            shapes[engine][:, 1:] = shapes[engine][:, 0][:, None] * shapes[engine][:, 1:]

        import matplotlib.pyplot as plt
        # now we want that all three engines have similar shapes
        nrows = int(0.5 * len(engines) * (len(engines) - 1))
        fig, axs = plt.subplots(figsize=(5 * 3, 4 * nrows), ncols=3, nrows=nrows, squeeze=False)
        # screw trying to figure out the math
        ax_row = 0
        for engine_i in range(len(engines)):
            for engine_j in range(engine_i + 1, len(engines)):
                for ax_i, ax in enumerate(axs[ax_row]):
                    x = shapes[engines[engine_i]][:, ax_i]
                    y = shapes[engines[engine_j]][:, ax_i]
                    min_val = np.min([x, y])
                    max_val = np.max([x, y])
                    ax.plot(x, y, 'bo', alpha=0.5)
                    ax.plot([min_val, max_val], [min_val, max_val], 'k--')
                    ax.set_xlabel(engines[engine_i])
                    ax.set_ylabel(engines[engine_j])
                ax.set_title(template)
                ax_row += 1
        plt.tight_layout()
        fig.savefig('wavefront_test/{0}.png'.format(template))

        # assert that available engines have same shape
        ax_row = 0
        nrows = int(0.5 * len(engines) * (len(engines) - 1))
        for engine_i in range(len(engines)):
            for engine_j in range(engine_i + 1, len(engines)):
                for ax_i in range(nrows):
                    x = shapes[engines[engine_i]][:, ax_i]
                    y = shapes[engines[engine_j]][:, ax_i]
                    diff = (x - y) / x
                    goal = np.zeros(len(diff))
                    np.testing.assert_almost_equal(diff, goal, decimal=3)
                ax_row += 1

def test_chisq():
    # we have some prior sense of how well the chisquare should perform to
    # flucuations in the parameters. We should be sensative to z0id to say 0.2
    # waves or something less than that. Let's make sure a '1 sigma' change of
    # that extent is detectable as actually '1 sigma'

    pass

def plot_star(star, vmin=None, vmax=None):
    # convenience function
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    IM = ax.imshow(star.image.array, interpolation='None', vmin=vmin, vmax=vmax)
    fig.colorbar(IM, ax=ax)

    return fig, ax

if __name__ == '__main__':
    test_init()
    # test_optical_engines()
    # test_fit()
    test_yaml()
    test_chisq()
