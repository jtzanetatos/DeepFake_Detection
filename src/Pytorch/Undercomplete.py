# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 23:12:28 2021

@author: iason
"""

import torch.nn as nn

# TODO: Change input/output image dimensions

class UAutoencoder(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        
        
        self.layers = nn.Sequential(
            # Encoder
            nn.Linear(in_features=img_size*img_size*3, out_features=256), # Input image (28*28 = 784)
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            
            # Decoder
            nn.Linear(in_features=16, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=img_size*img_size*3), # Output image (28*28 = 784)
            nn.ReLU())
    
    def forward(self, x):
        
        return self.layers(x)