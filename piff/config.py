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
.. module:: config
"""

from __future__ import print_function
import piff
import yaml

def setup_logger(verbosity=1, log_file=None):
    """Build a logger object to use for logging progress

    Note: This will update the verbosity if a previous call to setup_logger used a different
    value for verbose.  However, it will not update the handler to use a different log_file
    or switch between using a log_file and stdout.

    :param verbosity:   A number from 0-3 giving the level of verbosity to use. [default: 1]
    :param log_file:    A file name to which to output the logging information. [default: None]

    :returns: a logging.Logger instance
    """
    # Parse the integer verbosity level from the command line args into a logging_level string
    logging_levels = { 0: logging.CRITICAL,
                       1: logging.WARNING,
                       2: logging.INFO,
                       3: logging.DEBUG }
    logging_level = logging_levels[args.verbosity]
                                                                                                        # Setup logging to go to sys.stdout or (if requested) to an output file
    logger = logging.getLogger('piff')
    if len(logger.handlers) == 0:  # only add handler once!
        if log_file is None:
            handle = logging.StreamHandler()
        else:
            handle = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(message)s')  # Simple text output
        handle.setFormatter(formatter)
        logger.addHandler(handle)
    logger.setLevel(logging_level)

    return logger

def parse_variables(config, variables, logger):
    """Parse configuration variables and add them to the config dict

    The command line variables should be specified as key=value.
    The key string can include dots, such as interp.order=2, which means to set::

        config['interp']['order'] = 2

    :param config:      The configuration dict to wich to write the key,value pairs.
    :param varaibles:   A list of (typically command line) variables to parse.
    :param logger:      A logger object for logging debug info.
    """
    for v in variables:
        logger.debug('Parsing additional variable: %s',v)
        if '=' not in v:
            raise ValueError('Improper variable specificationi: %s.  Use field.item=value.'%v)
        key, value = v.split('=',1)
        try:
            # Use YAML parser to evaluate the string in case it is a list for instance.
            value = yaml.load(value)
        except:
            logger.debug('Unable to parse %s.  Treating it as a string.'%value)
        config[key] = value


def read_config(file_name):
    """Read a configuration dict from a file.

    :param file_name:   The file name from which the configuration dict should be read.
    """
    with open(file_name) as fin:
        config = yaml.load(f.read())
    return config


def piffify(config, logger):
    """Build a Piff model according to the specifications in a config dict.

    :param config:      The configuration file that defines how to build the model
    :param logger:      A logger object for logging progress
    """

    for key in ['input', 'output', 'model', 'interp']:
        if key not in config:
            raise ValueError("%s field is required in config dict"%key)

    extra_kwargs = {}

    # read in the input images
    images, stars, kwargs = piff.process_input(config['input'], logger)
    extre_kwargs.update(kwargs)

    # make a Model object to use for the individual stellar fitting
    model, kwargs = piff.process_model(config['model'], logger)
    extre_kwargs.update(kwargs)

    # make an Interp object to use for the interpolation
    interp, kwargs = piff.process_interp(config['interp'], logger)
    extre_kwargs.update(kwargs)

    # if given, make and Optics object to use as the prior information about the optics.
    if 'optics' in config:
        optics, kwargs = piff.process_optics(config['optics'], logger)
        extre_kwargs.update(kwargs)
    else:
        optics = None

    # build the PSF model
    psf = build_psf(images=images, stars=stars, model=model, interp=interp, optics=optics,
                    logger=logger, **extra_kwargs)

    # write it out to a file
    output, kwargs = piff.process_output(config['output'], logger)
    output.write(psf, **kwargs)
