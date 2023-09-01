#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

https://github.com/ThomasRieutord/MT-MLULC
"""
import os

_repopath_ = os.path.split(__path__[0])[0]

with open(os.path.join(_repopath_, "setup.py"), "r") as f:
    for l in f.readlines():
        if "version=" in l:
            __version__ = l.split('"')[1]
            break

del f, l, os
