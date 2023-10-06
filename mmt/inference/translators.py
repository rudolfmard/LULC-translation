#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Module with inference map translator
"""

import os
from mmt.inference import io
import onnx
import onnxruntime as ort

# BASE CLASSES
#==============
class OnnxModel:
    """Wrapper for inference from ONNX model"""
    def __init__(self, onnxfilename, inputname = None, outputname = None):
        self.onnxfilename = onnxfilename
        self.ort_session = ort.InferenceSession(onnxfilename, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        
        if outputname is None:
            self.outputname = os.path.basename(onnxfilename).split(".")[0].split("_")[-1]
        if inputname is None:
            self.inputname = os.path.basename(onnxfilename).split(".")[0].split("_")[-2]
            
        # Check the model
        onnx_model = onnx.load(onnxfilename)
        onnx.checker.check_model(onnx_model)
        
    def __call__(self, x):
        outputs = self.ort_session.run(
            [self.outputname],
            {self.inputname: x},
        )
        return outputs[0]

class MapTranslator:
    """Apply map translation models in inference mode"""
    def __init__(self, xp_name, lc_in, lc_out):
        self.xp_name = xp_name
        self.encoder = self.load_encoder(xp_name, lc_in)
        self.decoder = self.load_decoder(xp_name, lc_out)
        self.pos_enc = self.load_position_encoder(xp_name, lc_in)
        
    def __call__(self, data, coords = None):
        emb = self.encoder(data)
        if coords is not None:
            emb += self.pos_enc(coords)
        
        return self.decoder(emb)
    
    def load_encoder(self, xp_name, lc_in):
        raise NotImplementedError
        
    def load_decoder(self, xp_name, lc_out):
        raise NotImplementedError
        
    def load_position_encoder(self, xp_name, lc_in):
        raise NotImplementedError


class MapMixAndTranslate:
    """Take two input maps, mixes them and return a single output map.
    Apply map translation models in inference mode"""
    def __init__(self, xp_name, lc_in1, lc_in2, lc_out):
        self.xp_name = xp_name
        self.encoder1 = self.load_encoder(xp_name, lc_in1)
        self.encoder2 = self.load_encoder(xp_name, lc_in2)
        self.emb_mixer = self.load_embedding_mixer(xp_name)
        self.decoder = self.load_decoder(xp_name, lc_out)
        self.pos_enc = self.load_position_encoder(xp_name, lc_in)
        
    def __call__(self, data1, data2, coords = None):
        emb1 = self.encoder1(data1)
        emb2 = self.encoder2(data1)
        if coords is not None:
            emb1 += self.pos_enc(coords)
            emb2 += self.pos_enc(coords)
        
        emb = self.emb_mixer(emb1, emb2)
        
        return self.decoder(emb)
    
    def load_encoder(self, xp_name, lc_in):
        raise NotImplementedError
        
    def load_decoder(self, xp_name, lc_out):
        raise NotImplementedError
        
    def load_position_encoder(self, xp_name, lc_in):
        raise NotImplementedError
        
    def load_embedding_mixer(self, xp_name, lc_in):
        raise NotImplementedError

