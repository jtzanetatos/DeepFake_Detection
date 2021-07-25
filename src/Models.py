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
            nn.Flatten(),
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
    
    def forward(self, x):
        
        return self.layers(x)