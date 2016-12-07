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

from time import time

def test_init():
    load_decamwavefrontpsf()

def test_fit(engine='donutlib'):
    # fit only a couple parameters
    params = {'r0': 0.14,
              'z04d': 0.5,
              'z05d': 0.5,
              'z07d': -0.5,
              'z06x': 0.002,
              'z10d': -0.5,
              }
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
    for name, Stats in zip(['twodhist', 'whisker'], [piff.TwoDHistStats, piff.WhiskerStats]):
        stats = Stats(file_name='wavefront_test/fit_{0}.png'.format(name))
        stats.compute(psf, psf.stars)
        stats.write()

    # check fit
    import ipdb; ipdb.set_trace()

def test_full_fit():
    pass

def test_disk():
    pass

def test_yaml():
    pass

def generate_sample(params={}, n_samples=5000, engine='donutlib', seed=12345):

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
        star = piff.Star.makeTarget(x=icen, y=jcen, u=u, v=v, properties={'chipnum': chipnum}, stamp_size=24, scale=arcsecperpixel)
        stars.append(star)

    # get the focal positions
    stars = piff.des.DECamInfo().pixel_to_focalList(stars)

    # star = stars[0]
    # for engine_i, engine in enumerate(['regular', 'fast', 'donutlib']):
    #     # apply decamwavefrontpsf
    #     psf = load_decamwavefrontpsf(engine=engine)
    #     psf.update_psf_params(**params)
    #     # this is really slow?!
    #     # stars = [psf.drawStar(s) for s in stars]
    #     star = psf.drawStar(star)
    #     star.fit.center = (0, 0)
    #     print(engine)
    #     star.fit.params[0] = -1
    #     star.fit.params[1] = 0.5
    #     star.fit.params[3] = -1
    #     star.fit.params[5] = -0.5
    #     star = psf.drawStar(star)
    #     fig, ax = plot_star(star, vmin=0.00, vmax=0.05)
    #     ax.set_title(engine)
    # import matplotlib.pyplot as plt
    # plt.show()
    # raise Exception

    psf = load_decamwavefrontpsf(engine=engine)
    psf.update_psf_params(**params)
    time0 = time()
    stars = psf.drawStarList(stars)
    time1 = time()

    return stars, psf, time1 - time0

def load_decamwavefrontpsf(engine='galsim', do_pupil_plane_im=False):
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
        psf = piff.des.DECamWavefrontPSF(knn_file_name, knn_extname, pupil_plane_im, engine=engine, model_kwargs={'pupil_plane_scale': pupil_plane_scale})
    else:
        # obscuration slows things down tremendously. We are not particularly interested
        # in fidelity to DECam (vs fidelity to the code) for these tests, so set that to 0
        # psf = piff.des.DECamWavefrontPSF(knn_file_name, knn_extname, model_kwargs={'obscuration':0}, verbose=True, engine=engine)
        psf = piff.des.DECamWavefrontPSF(knn_file_name, knn_extname, verbose=True, engine=engine)

    return psf

def test_optical_engines():
    # we have rough expectations for how the model should measure moments, and
    # we need to make sure that messing with gsparams doesn't mess that up.
    # then we also need to test that the relationship is the same whether we
    # use the reduced GSParams or the default GSParams

    params = {'r0': 0.2,
              # 'z04d': 0.5,
              # 'z05d': 0.5,
              # 'z07d': -0.5,
              # 'z06x': 0.002,
              # 'z10d': -0.5,
              }
    n_samples = 50

    shapes = {}
    engines = ['donutlib', 'donutlib_again', 'donutlib_fast', 'donutlib_fast_scalefactor', 'donutlib_scalefactor', 'donutlib_old', 'galsim_fast', 'galsim']
    templates = ['des', 'lsst']
    import matplotlib.pyplot as plt
    for template in templates:
        for engine in engines:
            stars, wf, deltatime = generate_sample(params, n_samples, engine=engine)
            print('took {0:.2e} to drawStarList {1} stars for {2}'.format(deltatime, n_samples, engine))
            # get the shapes
            stars = [wf.model_comparer.fit(star) for star in stars]
            shapes[engine] = np.array([star.fit.params for star in stars])
            shapes[engine][:, 1:] = shapes[engine][:, 0][:, None] * shapes[engine][:, 1:]

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
        fig.savefig('{0}.pdf'.format(template))
    import ipdb; ipdb.set_trace()

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
    # test_init()
    test_optical_engines()
    # test_fit()
    # test_full_fit()
    # test_disk()
    # test_yaml()
    # test_fit()
