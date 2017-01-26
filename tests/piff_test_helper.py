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


# Some helper functions that mutliple test files might want to use

def which(program):
    """
    Mimic functionality of unix which command
    """
    import sys,os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    if sys.platform == "win32" and not program.endswith(".exe"):
        program += ".exe"

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None

def get_script_name(file_name):
    """
    Check if the file_name is in the path.  If not, prepend appropriate path to it.
    """
    import os
    if which(file_name) is not None:
        return file_name
    else:
        test_dir = os.path.split(os.path.realpath(__file__))[0]
        root_dir = os.path.split(test_dir)[0]
        script_dir = os.path.join(root_dir, 'scripts')
        exe_file_name = os.path.join(script_dir, file_name)
        print('Warning: The script %s is not in the path.'%file_name)
        print('         Using explcit path for the test:',exe_file_name)
        return exe_file_name

def timer(f):
    import functools

    @functools.wraps(f)
    def f2(*args, **kwargs):
        import time
        t0 = time.time()
        result = f(*args, **kwargs)
        t1 = time.time()
        fname = repr(f).split()[1]
        print('time for %s = %.2f' % (fname, t1-t0))
        return result
    return f2
