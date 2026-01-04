# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 01:29:16 2025

@author: zarkm
"""
import sys, os
print("sys.executable:", sys.executable)
print("sys.prefix:    ", sys.prefix)
print("CONDA env:     ", os.environ.get("CONDA_DEFAULT_ENV"))

import torch
print("CUDA available?   ", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)
print("GPU name:         ", torch.cuda.get_device_name(0))