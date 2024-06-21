#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Program to test import of the package.

Organisation of the package
-----------
mmt
├── agents
│   ├── __init__.py
│   ├── base.py
│   └── multiLULC.py
├── datasets
│   ├── __init__.py
│   ├── landcovers.py
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
│       ├── attention_autoencoder.py
│       └── universal_embedding.py
└── utils
    ├── __init__.py
    ├── aliases.py
    ├── config.py
    ├── misc.py
    ├── plt_utils.py
    └── scores.py
"""

# MODULES
# =======
import mmt
import mmt.agents
import mmt.datasets
import mmt.graphs
import mmt.graphs.models
import mmt.graphs.models.custom_layers
import mmt.utils
from mmt import _repopath_ as mmt_repopath

print(f"Package {mmt.__name__}-{mmt.__version__} from {mmt_repopath}")

# CLASSES
# =======

# Datasets
# --------
from mmt.datasets import landcover_to_landcover, landcovers
from mmt.datasets import transforms as mmt_transforms

mmt_transforms.EsawcTransform
landcovers.EcoclimapSG
landcover_to_landcover.LandcoverToLandcoverDataLoader

# Agents
# ------
from mmt.agents import base, multiLULC

base.BaseAgent
multiLULC.MultiLULCAgent

# Graphs
# ------
from mmt.graphs.models import universal_embedding
from mmt.graphs.models.custom_layers import up_block

universal_embedding.UnivEmb
up_block.Up

# Inference
# ------
from mmt.inference import io, translators

io.load_pytorch_model
translators.EsawcToEsgp

# Utils
# -----
from mmt.utils import config as utilconf
from mmt.utils import misc, plt_utils, scores

misc.memoize
scores.compute_confusion_matrix
plt_utils.plot_loss
utilconf.get_config


print("All imports passed successfully")
