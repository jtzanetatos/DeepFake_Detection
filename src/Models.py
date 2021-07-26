# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 23:12:28 2021

@author: iason
"""

import torch.nn as nn
import torch
import numpy as np

class UAutoencoder(nn.Module):
    def __init__(self, img_size, batch_size):
        super(UAutoencoder, self).__init__()
        
        self.inShape = img_size
        
        self.layers = nn.Sequential(
            # Encoder
            nn.Linear(in_features=self.inShape*self.inShape*3*batch_size,
                      out_features=128*batch_size),
            nn.ReLU(),
            nn.Linear(in_features=128*batch_size, out_features=64*batch_size),
            nn.ReLU(),
            nn.Linear(in_features=64*batch_size, out_features=32*batch_size),
            nn.ReLU(),
            nn.Linear(in_features=32*batch_size, out_features=16*batch_size),
            nn.ReLU(),
            
            # Decoder
            nn.Linear(in_features=16*batch_size, out_features=32*batch_size),
            nn.ReLU(),
            nn.Linear(in_features=32*batch_size, out_features=64*batch_size),
            nn.ReLU(),
            nn.Linear(in_features=64*batch_size, out_features=128*batch_size),
            nn.ReLU(),
            nn.Linear(in_features=128*batch_size,
                      out_features=self.inShape*self.inShape*3*batch_size),
            nn.Sigmoid())
    
    def forward(self, x):
        
        return self.layers(x)
    
    def predict(self, model, dataset, batch_size, img_shape):
        
        preds = np.zeros(len(dataset)*batch_size, dtype=object)
        
        idx = 0
        
        with torch.no_grad():
            for i, data in enumerate(dataset):
                
                if data.shape[0] > 1:
                    for j, batch in enumerate(data):
                        preds[idx] = self._output(model, data[j], img_shape)
                        idx += 1
                else:
                    preds[i] = self._output(model, data, img_shape)
        
        return preds
    
    def _output(self, model, data, img_shape):
        
        x_tensor = data.to('cuda').reshape(1, -1)
        pred = model(x_tensor)
        pred = pred.squeeze().cpu().numpy().round()
        
        return pred.T

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
    
    def predict(self, model, dataset, batch_size, img_shape):
        
        preds = np.zeros(len(dataset)*batch_size, dtype=object)
        
        idx = 0
        
        with torch.no_grad():
            for i, data in enumerate(dataset):
                
                if data.shape[0] > 1:
                    for j, batch in enumerate(data):
                        preds[idx] = self._output(model,
                                                  data[j].unsqueeze(0),
                                                  img_shape)
                        idx += 1
                else:
                    preds[i] = self._output(model, data, img_shape)
        
        return preds
    
    def _output(self, model, data, img_shape):
        
        x_tensor = data.to('cuda')
        pred = model(x_tensor)
        pred = pred.squeeze().cpu().numpy().round()
        
        return pred.T