#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

https://github.com/ThomasRieutord/MT-MLULC


 QUESTION
==========
What are the architectures of the tested models?
In particular, how many parameters do they have and what is their memory footprint?

 ANSWER
=========
The BASELINE model has 350031 trainable parameters and needs 20.20513536 GB to run
The CANDIDATE model has 355182 trainable parameters and needs 20.20956992 GB to run

The BASELINE model is the original MLCnet proposed by Baudoux et al. (2022)
The CANDIDATE model is the new version of model that includes attention layers

Reproducibility
---------------
python what_are_the_architectures_of_the_tested_models.py
"""
import os
import torch
from torchinfo import summary

from mmt.graphs.models import universal_embedding
from mmt.graphs.models import transformer_embedding
from mmt.utils import config as utilconf
from mmt import _repopath_ as mmt_repopath

# Configs
#---------
xp_name = "test_if_it_runs"
mlulcconfig, _ = utilconf.get_config_from_json(
    os.path.join(
        mmt_repopath,
        "configs",
        "universal_embedding.json",
    )
)
n_labels = 43
resize = None

baseline_model = universal_embedding.UnivEmb(
    in_channels = n_labels + 1,
    n_classes = n_labels + 1,
    softpos = mlulcconfig.softpos,
    number_feature_map = mlulcconfig.number_of_feature_map,
    embedding_dim = mlulcconfig.embedding_dim[0],
    memory_monger = mlulcconfig.memory_monger,
    up_mode = mlulcconfig.up_mode,
    num_groups = mlulcconfig.group_norm,
    decoder_depth = mlulcconfig.decoder_depth,
    mode = mlulcconfig.mode,
    resize = resize,
    cat=False,
    pooling_factors = mlulcconfig.pooling_factors,
    decoder_atrou = mlulcconfig.decoder_atrou,
)
baseline_model = torch.nn.Sequential(
    baseline_model.encoder,
    baseline_model.decoder,
)
candidate_model = transformer_embedding.TransformerEmbedding(
    in_channels = n_labels + 1,
    n_classes = n_labels + 1,
    softpos = mlulcconfig.softpos,
    number_feature_map = mlulcconfig.number_of_feature_map,
    embedding_dim = mlulcconfig.embedding_dim[0],
    memory_monger = mlulcconfig.memory_monger,
    up_mode = mlulcconfig.up_mode,
    num_groups = mlulcconfig.group_norm,
    decoder_depth = mlulcconfig.decoder_depth,
    mode = mlulcconfig.mode,
    resize = resize,
    cat=False,
    pooling_factors = mlulcconfig.pooling_factors,
    decoder_atrou = mlulcconfig.decoder_atrou,
)
candidate_model = torch.nn.Sequential(
    candidate_model.encoder,
    candidate_model.decoder,
)

x = torch.rand(15, n_labels + 1, 600, 600)

print("     Architecture of the BASELINE model")
baseline_summary = summary(baseline_model, x.shape)

print("\n     Architecture of the CANDIDATE model")
candidate_summary = summary(candidate_model, x.shape)

print(" ")
print(f"The BASELINE model has {baseline_summary.trainable_params} trainable parameters and needs {baseline_summary.total_output_bytes/10**9} GB to run")
print(f"The CANDIDATE model has {candidate_summary.trainable_params} trainable parameters and needs {candidate_summary.total_output_bytes/10**9} GB to run")
