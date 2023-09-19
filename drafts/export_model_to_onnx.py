#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Program to export MLCT-net to the ONNX format.
"""

import argparse
from mmt.inference import io

# Configs
#---------
# arg_parser = argparse.ArgumentParser(description="")
# arg_parser.add_argument('xp_name', default='aaunet2', help='Experiment name', nargs='?')
# arg_parser.add_argument('lc_in', default='esawc', help='Input land cover', nargs='?')
# arg_parser.add_argument('lc_out', default='esgp', help='Output land cover', nargs='?')
# args = arg_parser.parse_args()

# print(f"Exporting model to ONNX for xp_name={args.xp_name}, lc_in={}")
xp_name = "vanilla_no0"
lc_in = "esawc"
lc_out = "esgp"
onnxfilename = io.export_pytorch_to_onnx(xp_name, lc_in, lc_out)
print(f"ONNX file saved at: {onnxfilename}")
