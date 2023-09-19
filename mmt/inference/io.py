#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Module to load and export pretrained models for inference
"""

import os
import torch
from torch import nn
from mmt import _repopath_ as mmt_repopath
from mmt.graphs.models import (
    universal_embedding,
    transformer_embedding,
    attention_autoencoder
)
from mmt.datasets import landcover_to_landcover
from mmt.datasets import transforms as mmt_transforms
from mmt.utils import config as utilconf

patch_size_metres = landcover_to_landcover.patch_size_metres

def get_resize_from_mapname(mapname, config):
    
    if not mapname.endswith(".hdf5"):
        mapname += ".hdf5"
    
    try:
        # Old config
        n_px_emb = config.embedding_dim[1]
        model_type = config.model_type
    except:
        # New config
        n_px_emb = config.dimensions.n_px_embedding
        model_type = config.model.type
        
    resolution = landcover_to_landcover.resolution_dict[mapname]
    resize = int(n_px_emb * resolution/patch_size_metres)
    
    if model_type in ["transformer_embedding", "universal_embedding"] and resize == 1:
        resize = None
        
    return resize

def load_old_pytorch_model(xp_name, lc_in="esawc", lc_out="esgp"):
    """Return the pre-trained Pytorch model from the experiment `xp_name`"""
    try:
        config = utilconf.get_config(
            os.path.join(
                mmt_repopath,
                "experiments",
                xp_name,
                "logs",
                "config.json",
            )
        )
    except:
        raise RuntimeError(f"Unable to load config from xp {xp_name}. Maybe try load_new_pytorch_model?")
        
    checkpoint_path = os.path.join(
        mmt_repopath,
        "experiments",
        xp_name,
        "checkpoints",
        "model_best.pth.tar",
    )
    assert os.path.isfile(checkpoint_path), f"No checkpoint found at {checkpoint_path}"

    res_in = landcover_to_landcover.resolution_dict[lc_in + ".hdf5"]
    res_out = landcover_to_landcover.resolution_dict[lc_out + ".hdf5"]
    n_channels_in = len(landcover_to_landcover.label_dict[lc_in + ".hdf5"]) + 1
    n_channels_out = len(landcover_to_landcover.label_dict[lc_out + ".hdf5"]) + 1
    n_channels_emb = config.embedding_dim[0]
    n_px_emb = config.embedding_dim[1]

    if config.model_type == "transformer_embedding":
        try:
            EncDec = getattr(transformer_embedding, config.model_name)
        except AttributeError:
            print("Config doesn't have a model_name attribute. Loaded transformer_embedding.TransformerEmbedding")
            EncDec = transformer_embedding.TransformerEmbedding
        
        autoenc_in = EncDec(
            n_channels_in,
            n_channels_in,
            mul=config.mul,
            softpos=config.softpos,
            number_feature_map=config.number_of_feature_map,
            embedding_dim=n_channels_emb,
            memory_monger=config.memory_monger,
            up_mode=config.up_mode,
            num_groups=config.group_norm,
            decoder_depth=config.decoder_depth,
            mode=config.mode,
            resize = get_resize_from_mapname(lc_in, config),
            cat=False,
            pooling_factors=config.pooling_factors,
            decoder_atrou=config.decoder_atrou,
        )
        autoenc_out = EncDec(
            n_channels_out,
            n_channels_out,
            mul=config.mul,
            softpos=config.softpos,
            number_feature_map=config.number_of_feature_map,
            embedding_dim=config.embedding_dim[0],
            memory_monger=config.memory_monger,
            up_mode=config.up_mode,
            num_groups=config.group_norm,
            decoder_depth=config.decoder_depth,
            mode=config.mode,
            resize = get_resize_from_mapname(lc_out, config),
            cat=False,
            pooling_factors=config.pooling_factors,
            decoder_atrou=config.decoder_atrou,
        )
    elif config.model_type == "universal_embedding":
        try:
            EncDec = getattr(universal_embedding, config.model_name)
        except AttributeError:
            print("Config doesn't have a model_name attribute. Loaded universal_embedding.UnivEmb")
            EncDec = universal_embedding.UnivEmb
                
        autoenc_in = EncDec(
            n_channels_in,
            n_channels_in,
            mul=config.mul,
            softpos=config.softpos,
            number_feature_map=config.number_of_feature_map,
            embedding_dim=n_channels_emb,
            memory_monger=config.memory_monger,
            up_mode=config.up_mode,
            num_groups=config.group_norm,
            decoder_depth=config.decoder_depth,
            mode=config.mode,
            resize = get_resize_from_mapname(lc_in, config),
            cat=False,
            pooling_factors=config.pooling_factors,
            decoder_atrou=config.decoder_atrou,
        )
        autoenc_out = EncDec(
            n_channels_out,
            n_channels_out,
            mul=config.mul,
            softpos=config.softpos,
            number_feature_map=config.number_of_feature_map,
            embedding_dim=config.embedding_dim[0],
            memory_monger=config.memory_monger,
            up_mode=config.up_mode,
            num_groups=config.group_norm,
            decoder_depth=config.decoder_depth,
            mode=config.mode,
            resize = get_resize_from_mapname(lc_out, config),
            cat=False,
            pooling_factors=config.pooling_factors,
            decoder_atrou=config.decoder_atrou,
        )
    elif config.model_type == "attention_autoencoder":
        try:
            EncDec = getattr(attention_autoencoder, config.model_name)
        except AttributeError:
            print("Config doesn't have a model_name attribute. Loaded attention_autoencoder.AttentionAutoEncoderSC")
            EncDec = attention_autoencoder.AttentionAutoEncoderSC
        
        autoenc_in = EncDec(
            n_channels_in,
            n_channels_in,
            h_channels=config.number_of_feature_map,
            emb_channels=config.embedding_dim[0],
            emb_size_ratio=int(config.embedding_dim[1] / 10),
            resize=get_resize_from_mapname(lc_in, config),
        )
        autoenc_out = EncDec(
            n_channels_out,
            n_channels_out,
            h_channels=config.number_of_feature_map,
            emb_channels=config.embedding_dim[0],
            emb_size_ratio=int(config.embedding_dim[1] / 10),
            resize=get_resize_from_mapname(lc_out, config),
        )
    else:
        raise ValueError(f"Unknown model_type = {config.model_type}. Please change config to one among ['transformer_embedding', 'universal_embedding', 'attention_autoencoder']")
    
    checkpoint = torch.load(checkpoint_path)
    
    autoenc_in.load_state_dict(checkpoint[f"encoder_state_dict_{lc_in}.hdf5"])
    autoenc_out.load_state_dict(checkpoint[f"encoder_state_dict_{lc_out}.hdf5"])
    
    model = nn.Sequential(
        autoenc_in.encoder,
        autoenc_out.decoder
    )
    
    return model
    
def load_pytorch_model(xp_name, lc_in="esawc", lc_out="esgp"):
    """Return the pre-trained Pytorch model from the experiment `xp_name`"""
    try:
        config = utilconf.get_config(
            os.path.join(
                mmt_repopath,
                "experiments",
                xp_name,
                "logs",
                "config.yaml",
            )
        )
    except:
        print("Loading old JSON config")
        config = utilconf.get_config(
            os.path.join(
                mmt_repopath,
                "experiments",
                xp_name,
                "logs",
                "config.json",
            )
        )
        
    checkpoint_path = os.path.join(
        mmt_repopath,
        "experiments",
        xp_name,
        "checkpoints",
        "model_best.pth.tar",
    )
    assert os.path.isfile(checkpoint_path), f"No checkpoint found at {checkpoint_path}"

    res_in = landcover_to_landcover.resolution_dict[lc_in + ".hdf5"]
    res_out = landcover_to_landcover.resolution_dict[lc_out + ".hdf5"]
    n_channels_in = len(landcover_to_landcover.label_dict[lc_in + ".hdf5"]) + 1
    n_channels_out = len(landcover_to_landcover.label_dict[lc_out + ".hdf5"]) + 1
    
    if config.model.type == "transformer_embedding":
        EncDec = getattr(transformer_embedding, config.model.name)
    elif config.model.type == "universal_embedding":
        EncDec = getattr(universal_embedding, config.model.name)
    elif config.model.type == "attention_autoencoder":
        EncDec = getattr(attention_autoencoder, config.model.name)
    else:
        raise ValueError(f"Unknown model.type = {config.model.type}. Please change config to one among ['transformer_embedding', 'universal_embedding', 'attention_autoencoder']")
    
    autoenc_in = EncDec(
        n_channels_in,
        n_channels_in,
        resize = get_resize_from_mapname(lc_in, config),
        n_channels_hiddenlay = config.dimensions.n_channels_hiddenlay,
        n_channels_embedding = config.dimensions.n_channels_embedding,
        **config.model.params
    )
    autoenc_out = EncDec(
        n_channels_out,
        n_channels_out,
        resize = get_resize_from_mapname(lc_out, config),
        n_channels_hiddenlay = config.dimensions.n_channels_hiddenlay,
        n_channels_embedding = config.dimensions.n_channels_embedding,
        **config.model.params
    )
    
    checkpoint = torch.load(checkpoint_path)
    
    autoenc_in.load_state_dict(checkpoint[f"encoder_state_dict_{lc_in}.hdf5"])
    autoenc_out.load_state_dict(checkpoint[f"encoder_state_dict_{lc_out}.hdf5"])
    
    model = nn.Sequential(
        autoenc_in.encoder,
        autoenc_out.decoder
    )
    
    return model

def export_pytorch_to_onnx(xp_name, lc_in="esawc", lc_out="esgp", onnxfilename = "[default].onnx"):
    """Load the Pytorch model and export it to the ONNX format"""
    
    if "[default]" in onnxfilename:
        onnxfilename = onnxfilename.replace(
            "[default]",
            os.path.join(
                mmt_repopath, "experiments", xp_name, "checkpoints", f"{lc_in}_to_{lc_out}"
            )
        )
    
    model = load_pytorch_model(xp_name, lc_in=lc_in, lc_out=lc_out)
    
    res_in = landcover_to_landcover.resolution_dict[lc_in + ".hdf5"]
    n_channels_in = len(landcover_to_landcover.label_dict[lc_in + ".hdf5"]) + 1
    n_px = patch_size_metres // res_in
    
    x = torch.rand(1, n_channels_in, n_px, n_px)
    torch.onnx.export(model, x, onnxfilename, input_names=[lc_in], output_names=[lc_out])
    
    return onnxfilename
