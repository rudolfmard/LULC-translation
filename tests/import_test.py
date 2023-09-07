#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Program to test import of the package.

Organisation of the package
-----------
mmt
├── agents
│   ├── base.py
│   ├── __init__.py
│   ├── multiLULC.py
│   └── TranslatingUnet_vf.py
├── datasets
│   ├── __init__.py
│   ├── landcover_to_landcover.py
│   └── transforms.py
├── graphs
│   ├── __init__.py
│   └── models
│       ├── custom_layers
│       │   ├── double_conv.py
│       │   ├── down_block.py
│       │   ├── __init__.py
│       │   └── up_block.py
│       ├── __init__.py
│       ├── translating_unet.py
│       └── universal_embedding.py
└── utils
    ├── config.py
    ├── dirs.py
    ├── image_type.py
    ├── __init__.py
    ├── misc.py
    ├── plt_utils.py
    └── tensorboardx_utils.py
"""

import mmt
import mmt.agents
import mmt.datasets
import mmt.utils
import mmt.graphs
import mmt.graphs.models
import mmt.graphs.models.custom_layers
from mmt import _repopath_ as mmt_repopath

from mmt.datasets import landcover_to_landcover
landcover_to_landcover.LandcoverToLandcoverDataLoader

from mmt.agents import base
base.BaseAgent
from mmt.agents import multiLULC
multiLULC.MultiLULCAgent

from mmt.graphs.models import universal_embedding
universal_embedding.UnivEmb

from mmt.graphs.models.custom_layers import up_block
up_block.Up

from mmt.utils import plt_utils
plt_utils.plt_loss2
from mmt.utils import config as utilconf
utilconf.get_config_from_json

print("All imports passed successfully")
print(f"Package {mmt.__name__}-{mmt.__version__} from {mmt._repopath_}")
