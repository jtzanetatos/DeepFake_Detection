# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 23:12:28 2021

@author: iason
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm

class UAutoencoder(nn.Module):
    def __init__(self, img_size, batch_size):
        super(UAutoencoder, self).__init__()
        
        self.inShape = img_size
        self.batch_size = batch_size
        self.epochIdx = 0
        
        self.layers = nn.Sequential(
            # Encoder
            nn.Linear(in_features=self.inShape*self.inShape*self.batch_size,
                      out_features=128*self.batch_size),
            nn.ReLU(),
            nn.Linear(in_features=128*self.batch_size, out_features=64*self.batch_size),
            nn.ReLU(),
            nn.Linear(in_features=64*self.batch_size, out_features=32*self.batch_size),
            nn.ReLU(),
            nn.Linear(in_features=32*self.batch_size, out_features=16*self.batch_size),
            nn.ReLU(),
            
            # Decoder
            nn.Linear(in_features=16*self.batch_size, out_features=32*self.batch_size),
            nn.ReLU(),
            nn.Linear(in_features=32*self.batch_size, out_features=64*self.batch_size),
            nn.ReLU(),
            nn.Linear(in_features=64*self.batch_size, out_features=128*self.batch_size),
            nn.ReLU(),
            nn.Linear(in_features=128*self.batch_size,
                      out_features=self.inShape*self.inShape*self.batch_size),
            nn.Sigmoid()
            )
    
    def forward(self, x):
        
        return self.layers(x)
    
    def trainModel(self, metrics, optimizer, dataset, epochs):
        
        self.train()
        
        currLoss = 0.0
        
        train_set = tqdm(dataset)
        
        for i, data in enumerate(train_set):
            
            if data.shape[0] > 1:
                for j, batch in enumerate(data):
                    for k in range(data.shape[1]):
                        inputs = data[j, k].view(1, -1)
                        inputs = inputs.to('cuda') if torch.cuda.is_available() else 'cpu'
                        
                        optimizer.zero_grad()
                        
                        outputs = self.layers(inputs)
                        
                        loss = metrics(outputs, inputs)
                        
                        loss.backward()
                        
                        optimizer.step()
                        
                        currLoss += loss.item()
            else:
                for k in range(data.shape[1]):
                    inputs = data[0, k].view(1, -1)
                    inputs = inputs.to('cuda') if torch.cuda.is_available() else 'cpu'
                    optimizer.zero_grad()
                    
                    outputs = self.layers(inputs)
                    
                    loss = metrics(outputs, inputs)
                    
                    loss.backward()
                    
                    optimizer.step()
                    
                    currLoss += loss.item()
            train_set.set_description(f"Epoch [{self.epochIdx+1}/{epochs}]")
            train_set.set_postfix(loss=loss.item())
            
        self.epochIdx += 1
        
        return currLoss / len(dataset)
    
    def evaluate(self, metrics, dataset, img_shape):
        
        preds = np.zeros(len(dataset)*self.batch_size, dtype=object)
        
        idx = 0
        
        loss = 0.0
        
        val_set = tqdm(dataset)
        
        with torch.no_grad():
            for i, data in enumerate(val_set):
                
                if data.shape[0] > 1:
                    for j, batch in enumerate(data):
                        tempPred = np.zeros((img_shape[0]*img_shape[1], img_shape[2]))
                        for k in range(img_shape[2]):
                            tempPred[:,k], currLoss = self._output(metrics, data[j, k], img_shape)
                            
                            loss += currLoss
                        preds[idx] = tempPred
                        idx += 1
                else:
                    tempPred = np.zeros((img_shape[0]*img_shape[1], img_shape[2]))
                    for k in range(img_shape[2]):
                        tempPred[:,k], currLoss = self._output(metrics, data[0, k], img_shape)
                        loss += currLoss
                    preds[idx] = tempPred.reshape(img_shape)
                    idx += 1
                val_set.set_description("Validation")
                val_set.set_postfix(loss=loss)
        
        return preds, loss.item()
    
    def _output(self, metrics, data, img_shape):
        
        x_tensor = data.view(1, -1).to('cuda') if torch.cuda.is_available() else 'cpu'
        pred = self.layers(x_tensor)
        loss = metrics(pred, x_tensor)
        pred = pred.squeeze().cpu().numpy().round()
        
        return pred.T, loss


class SparceAutoencoder(nn.Module):
    
    def __init__(self, img_size, batch_size, sparcity='l1', reg=0.001):
        super(SparceAutoencoder, self).__init__()
        self.reg = reg
        self.inShape = img_size
        self.sparcity = sparcity
        self.batch_size = batch_size
        
        self.epochIdx = 0
        
        # Encoder
        self.enc1 = nn.Linear(in_features=self.inShape*self.inShape*self.batch_size,
                  out_features=128*self.batch_size)
        
        self.enc2 = nn.Linear(in_features=128*self.batch_size, out_features=64*self.batch_size)
        
        self.enc3 = nn.Linear(in_features=64*self.batch_size, out_features=32*self.batch_size)
        
        self.enc4 = nn.Linear(in_features=32*self.batch_size, out_features=16*self.batch_size)
        
        
        # Decoder
        self.dec1 = nn.Linear(in_features=16*self.batch_size, out_features=32*self.batch_size)
        
        self.dec2 = nn.Linear(in_features=32*self.batch_size, out_features=64*self.batch_size)
        
        self.dec3 = nn.Linear(in_features=64*self.batch_size, out_features=128*self.batch_size)
        
        self.dec4 = nn.Linear(in_features=128*self.batch_size,
                  out_features=self.inShape*self.inShape*self.batch_size)
        
    def forward(self, x):
        
        # Encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        
        # Decoding
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        
        return torch.sigmoid(self.dec4(x))
    
    def _sparceLoss(self, data):
        
        loss = 0.0
        
        layers = list(self.children())
        
        for i in range(len(layers)):
            data = F.relu(layers[i](data))
            loss += torch.mean(torch.abs(data))
        
        return loss
    
    def trainModel(self, metrics, optimizer, dataset, epochs):
        
        self.train()
        
        currLoss = 0.0
        
        train_set = tqdm(dataset)
        
        for i, data in enumerate(train_set):
            
            if data.shape[0] > 1:
                for j, batch in enumerate(data):
                    for k in range(data.shape[1]):
                        inputs = data[j, k].view(1, -1)
                        inputs = inputs.to('cuda') if torch.cuda.is_available() else 'cpu'
                        optimizer.zero_grad()
                        
                        outputs = self.forward(inputs)
                        
                        loss = metrics(outputs, inputs)
                        
                        if self.sparcity.lower() == 'l1':
                            l1 = self._sparceLoss(inputs)
                            
                            loss += self.reg * l1
                        
                        loss.backward()
                        
                        optimizer.step()
                        
                        currLoss += loss.item()
            else:
                for k in range(data.shape[1]):
                    inputs = data[0, k].view(1, -1)
                    inputs = inputs.to('cuda') if torch.cuda.is_available() else 'cpu'
                    optimizer.zero_grad()
                    
                    outputs = self.forward(inputs)
                    
                    loss = metrics(outputs, inputs)
                    
                    if self.sparcity.lower() == 'l1':
                        l1 = self._sparceLoss(inputs)
                        
                        loss += self.reg * l1
                    
                    loss.backward()
                    
                    optimizer.step()
                    
                    currLoss += loss.item()
            train_set.set_description(f"Epoch [{self.epochIdx+1}/{epochs}]")
            train_set.set_postfix(loss=loss.item())
            
        self.epochIdx += 1
        
        return currLoss / len(dataset)
    
    def evaluate(self, metrics, dataset, img_shape):
        
        preds = np.zeros(len(dataset)*self.batch_size, dtype=object)
        
        idx = 0
        
        loss = 0.0
        
        val_set = tqdm(dataset)
        
        with torch.no_grad():
            for i, data in enumerate(val_set):
                
                if data.shape[0] > 1:
                    for j, batch in enumerate(data):
                        tempPred = np.zeros((img_shape[0]*img_shape[1], img_shape[2]))
                        for k in range(img_shape[2]):
                            tempPred[:,k], currLoss = self._output(metrics, data[j, k], img_shape)
                            
                            loss += currLoss
                        preds[idx] = tempPred
                        idx += 1
                else:
                    tempPred = np.zeros((img_shape[0]*img_shape[1], img_shape[2]))
                    for k in range(img_shape[2]):
                        tempPred[:,k], currLoss = self._output(metrics, data[0, k], img_shape)
                        loss += currLoss
                    preds[idx] = tempPred.reshape(img_shape)
                    idx += 1
                val_set.set_description("Validation")
                val_set.set_postfix(loss=loss)
        
        return preds, loss.item()
    
    def _output(self, metrics, data, img_shape):
        
        x_tensor = data.view(1, -1).to('cuda') if torch.cuda.is_available() else 'cpu'
        pred = self.forward(x_tensor)
        loss = metrics(pred, x_tensor)
        pred = pred.squeeze().cpu().numpy().round()
        
        return pred.T, loss

class ConvAutoencoder(nn.Module):
    def __init__(self, batch_size):
        super(ConvAutoencoder, self).__init__()
        
        self.batch_size = batch_size
        self.epochIdx = 0
        
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
            nn.ReLU()
            )
    
    def forward(self, x):
        
        return self.layers(x)
    
    def trainModel(self, metrics, optimizer, dataset, epochs):
        
        self.train()
        
        currLoss = 0.0
        
        train_set = tqdm(dataset)
        
        for i, data in enumerate(train_set):
            
            
            inputs = data.to('cuda') if torch.cuda.is_available() else 'cpu'
            optimizer.zero_grad()
            
            outputs = self.layers(inputs)
            
            loss = metrics(outputs, inputs)
            
            loss.backward()
            
            optimizer.step()
            
            currLoss += loss.item()
            
            train_set.set_description(f"Epoch [{self.epochIdx+1}/{epochs}]")
            train_set.set_postfix(loss=loss.item())
            
        self.epochIdx += 1
        
        return currLoss / len(dataset)
    
    def evaluate(self, metrics, dataset, img_shape):
        
        self.eval()
        
        preds = np.zeros(len(dataset)*self.batch_size, dtype=object)
        
        idx = 0
        
        loss = 0.0
        
        val_set = tqdm(dataset)
        
        with torch.no_grad():
            for i, data in enumerate(val_set):
                
                if data.shape[0] > 1:
                    for j, batch in enumerate(data):
                        preds[idx], currLoss = self._output(metrics, data[j].unsqueeze(0),
                                                  img_shape)
                        idx += 1
                        
                        loss += currLoss
                else:
                    preds[i], currLoss = self._output(metrics, data, img_shape)
                    
                    loss += currLoss
                val_set.set_description("Validation")
                val_set.set_postfix(loss=loss)
        
        return preds, loss
    
    def _output(self, metrics, data, img_shape):
        
        x_tensor = data.to('cuda') if torch.cuda.is_available() else 'cpu'
        pred = self.layers(x_tensor)
        loss = metrics(pred, x_tensor)
        pred = pred.squeeze().cpu().numpy().round()
        
        return pred.T, loss.item()