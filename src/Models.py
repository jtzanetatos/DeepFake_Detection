# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 23:12:28 2021

@author: iason
"""

import torch.nn as nn
import torch
import numpy as np

# TODO: Change input/output image dimensions

class UAutoencoder(nn.Module):
    def __init__(self, img_size):
        super(UAutoencoder, self).__init__()
        
        self.inShape = img_size
        
        self.layers = nn.Sequential(
            # Encoder
            # nn.Flatten(),
            nn.Linear(in_features=self.inShape*self.inShape*3, out_features=256),
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
            nn.Linear(in_features=256, out_features=self.inShape*self.inShape*3),
            nn.Sigmoid())
    
    def forward(self, x):
        
        return self.layers(x)
    
    def predict(self, dataset, img_shape):
        best_model = torch.load('./pytorch_best_model.pth')
    
        preds = np.zeros(len(dataset), dtype=object)
        
        with torch.no_grad():
            for i, data in enumerate(dataset):
                
                x_tensor = data.to('cuda').reshape(1, -1)
                pr_mask = best_model(x_tensor)
                pr_mask = pr_mask.squeeze().cpu().numpy().round()
                
                # Cast to uint8
                pr_mask = np.uint8(pr_mask.reshape(img_shape)*255)
                
                preds[i] = pr_mask
        
        return preds

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        self.layers = nn.Sequential(
            # Encoder
            nn.Conv2d(in_channels=3,
                      out_channels=16,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=16,
                      out_channels=4,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            # Decoder
            nn.ConvTranspose2d(in_channels=4,
                      out_channels=16,
                      kernel_size=2,
                      stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16,
                      out_channels=3,
                      kernel_size=2,
                      stride=2),
            nn.Sigmoid()
            )
    
    def forward(self, x):
        
        return self.layers(x)
    
    def predict(self, dataset, img_shape):
        best_model = torch.load('./pytorch_best_model.pth')
        
        preds = np.zeros(len(dataset), dtype=object)
        
        with torch.no_grad():
            for i, data in enumerate(dataset):
                
                x_tensor = data.to('cuda')
                pr_mask = best_model(x_tensor)
                pr_mask = pr_mask.squeeze().cpu().numpy().round()
                
                # Cast to uint8
                pr_mask = np.uint8(pr_mask.reshape(img_shape)*255)
                
                preds[i] = pr_mask
        
        return preds