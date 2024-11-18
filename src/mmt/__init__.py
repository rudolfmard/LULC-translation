#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

https://github.com/ThomasRieutord/MT-MLULC
"""
import os

_repopath_ = os.path.dirname(os.path.dirname(__path__[0]))

with open(os.path.join(_repopath_, "pyproject.toml"), "r") as f:
    for l in f.readlines():
        if "version =" in l:
            __version__ = l.split('"')[1]
            break

del f, l, os
