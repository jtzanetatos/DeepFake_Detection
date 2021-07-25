#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 14:23:57 2021

@author: iason
"""

import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        
        self.layers = nn.Sequential(
            # Encoder
            nn.Conv2d(in_channels=img_size*img_size*3,
                      out_channels=256,
                      kernel_size=(9, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.Conv2d(in_channels=256,
                      out_channels=128,
                      kernel_size=(9,9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.Conv2d(in_channels=128,
                      out_channels=64,
                      kernel_size=(9,9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.Conv2d(in_channels=64,
                      out_channels=32,
                      kernel_size=(9,9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
            
            # Decoder
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=(9,9)),
            nn.ReLU(),
            nn.MaxUnpool2d(kernel_size=(3,3)),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(9,9)),
            nn.ReLU(),
            nn.MaxUnpool2d(kernel_size=(3,3)),
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(9,9)),
            nn.ReLU(),
            nn.MaxUnpool2d(kernel_size=(3,3)),
            nn.Conv2d(in_channels=256,
                      out_channels=img_size*img_size*3,
                      kernel_size=(9, 9)),
            nn.ReLU()
            )