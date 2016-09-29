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

def test_init():
    load_decamwavefrontpsf()

def test_fit():
    # fit only a couple parameters
    params = {'r0': 0.2,
              'z05d': 0.2,
              }
    n_samples = 50
    psf = load_decamwavefrontpsf()
    stars, wf = generate_sample(params, n_samples)

    # speed things up by fixing the other keys
    for key in psf.minuit_kwargs:
        if 'fix' in key:
            if key.split('fix_')[-1] in params.keys():
                psf.minuit_kwargs[key] = False
            else:
                psf.minuit_kwargs[key] = True

    psf.fit(stars, None, None)

    # check fit
    import ipdb; ipdb.set_trace()

    return

def test_full_fit():
    pass

def test_disk():
    pass

def test_yaml():
    pass

def generate_sample(params={}, n_samples=5000):
    chipnums = np.random.randint(1, 63, n_samples)
    icens = np.random.randint(1, 2048, n_samples)
    jcens = np.random.randint(1, 4096, n_samples)

    # create stars
    stars = []
    arcsecperpixel = 0.263 # arcsec per pixel
    decaminfo = piff.des.decaminfo.DECamInfo()
    for icen, jcen, chipnum in zip(icens, jcens, chipnums):
        # convert icen to u based on chipnums.
        # focal plane center: (u,v) = (0,0)
        # get focal coords
        xpos, ypos = decaminfo.getPosition_chipnum(chipnum, icen, jcen)
        # convert from mm to uv
        u = xpos / decaminfo.mmperpixel * arcsecperpixel
        v = ypos / decaminfo.mmperpixel * arcsecperpixel

        # we make the star smaller to speed things up
        star = piff.Star.makeTarget(x=icen, y=jcen, u=u, v=v, properties={'chipnum': chipnum}, stamp_size=24, scale=arcsecperpixel)
        stars.append(star)

    # get the focal positions
    stars = piff.des.DECamInfo().pixel_to_focalList(stars)

    # apply decamwavefrontpsf
    psf = load_decamwavefrontpsf()
    psf.update_psf_params(**params)

    # this is really slow?!
    stars = [psf.drawStar(s) for s in stars]
    return stars, psf

def load_decamwavefrontpsf():
    knn_file_name = 'wavefront_test/Science-20121120s1-v20i2.fits'
    knn_extname = 'Science-20121120s1-v20i2'
    # pupil plane image is not working yet? or I need to specify scale...
    # pupil_plane_im = 'optics_test/DECam_pupil_512.fits'
    # psf = piff.des.DECamWavefrontPSF(knn_file_name, knn_extname, pupil_plane_im)
    # obscuration slows things down tremendously. We are not particularly interested
    # in fidelity to DECam (vs fidelity to the code) for these tests, so set that to 0
    psf = piff.des.DECamWavefrontPSF(knn_file_name, knn_extname, model_kwargs={'obscuration':0}, verbose=True)

    return psf

def plot_star(star):
    # convenience function
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(star.image.array)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    # test_init()
    # test_fit()
    # test_full_fit()
    # test_disk()
    # test_yaml()
    psf, stars, wf = test_fit()
