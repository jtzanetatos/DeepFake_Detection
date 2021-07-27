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
                        inputs = data[:, k].reshape(1, -1)
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
                    raise NotImplementedError
                    # TODO: Fix batch processing
                    # TODO: Code fix here:
                    # for j, batch in enumerate(data):
                    #     tempPred = np.zeros((img_shape[0]*img_shape[1]*self.batch_size,
                    #                          img_shape[2]))
                    #     for k in range(img_shape[2]):
                    #         tempPred[:,k], currLoss = self._output(metrics, data[:, k])
                            
                    #         loss += currLoss
                    #         # Iterate over batches
                    #         for n in range(self.batch_size):
                    #             if n == 0:
                    #                 predOut = tempPred[:img_shape[0]*img_shape[1]*(j+1), k]
                    #             else:
                    #                 predOut = tempPred[img_shape[0]*img_shape[1]*j:
                    #                               img_shape[0]*img_shape[1]*(j+1), k]
                    #             if k == 0:
                    #                 preds[idx+n] = predOut
                    #             elif k > 1:
                    #                 preds[idx+n] = np.dstack((preds[idx+n],
                    #                                           predOut.reshape(self.inShape, self.inShape)))
                    #             else:
                    #                 preds[idx+n] = np.dstack((preds[idx+n],
                    #                                           predOut)).reshape(self.inShape, self.inShape, 2)
                    #     idx += self.batch_size
                else:
                    tempPred = np.zeros((img_shape[0]*img_shape[1], img_shape[2]))
                    for k in range(img_shape[2]):
                        tempPred[:,k], currLoss = self._output(metrics, data[0, k])
                        loss += currLoss
                    preds[idx] = tempPred.reshape(img_shape)
                    idx += 1
                val_set.set_description("Validation")
                val_set.set_postfix(loss=loss.item())
        
        return preds, loss.item()
    
    def _output(self, metrics, data):
        
        if self.batch_size > 1:
            x_tensor = data.reshape(1, -1).to('cuda') if torch.cuda.is_available() else 'cpu'
        else:
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
    
    def _klDiv(rho, rho_hat):
        rho_hat = torch.mean(torch.sigmoid(rho_hat), 1)
        rho = torch.tensor([rho] * len(rho_hat)).to('cuda') if torch.cuda.is_available() else 'cpu'
        return torch.sum(rho * torch.log(rho/rho_hat) + (1-rho) * torch.log((1-rho)/(1-rho_hat)))
    
    def _sparceLoss(self, data, rho=None):
        
        loss = 0.0
        
        layers = list(self.children())
        
        # L1 Regularization
        if self.sparcity.lower() == 'l1':
            for i in range(len(layers)):
                data = F.relu(layers[i](data))
                loss += torch.mean(torch.abs(data))
        # KL Divergence
        else:
            for i in range(len(layers)):
                data = layers[i](data)
                loss += self._klDiv(rho, data)
        
        return loss
    
    def trainModel(self, metrics, optimizer, dataset, epochs, rho=None):
        
        self.train()
        
        currLoss = 0.0
        
        train_set = tqdm(dataset)
        
        for i, data in enumerate(train_set):
            
            if data.shape[0] > 1:
                for j, batch in enumerate(data):
                    for k in range(data.shape[1]):
                        inputs = data[:, k].reshape(1, -1)
                        inputs = inputs.to('cuda') if torch.cuda.is_available() else 'cpu'
                        optimizer.zero_grad()
                        
                        outputs = self.forward(inputs)
                        
                        loss = metrics(outputs, inputs)
                        
                        if self.sparcity.lower() == 'l1':
                            l1 = self._sparceLoss(inputs)
                            
                            loss += self.reg * l1
                        else:
                            kl = self._sparceLoss(inputs, rho)
                            
                            loss += self.reg * kl
                        
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
                    raise NotImplementedError
                    # TODO: Output code here
                else:
                    tempPred = np.zeros((img_shape[0]*img_shape[1], img_shape[2]))
                    for k in range(img_shape[2]):
                        tempPred[:,k], currLoss = self._output(metrics, data[0, k])
                        loss += currLoss
                    preds[idx] = tempPred.reshape(img_shape)
                    idx += 1
                val_set.set_description("Validation")
                val_set.set_postfix(loss=loss.item())
        
        return preds, loss.item()
    
    def _output(self, metrics, data):
        
        x_tensor = data.view(1, -1).to('cuda') if torch.cuda.is_available() else 'cpu'
        pred = self.forward(x_tensor)
        loss = metrics(pred, x_tensor)
        pred = pred.squeeze().cpu().numpy().round()
        
        return pred.T, loss

class VarAutoencoder(nn.Module):
    
    def __init__(self, img_size, batch_size, features=16):
        super(VarAutoencoder, self).__init__()
        
        self.img_size = img_size
        self.batch_size = batch_size
        self.features = features
        
        # Encoder
        self.enc1 = nn.Linear(in_features=self.img_size*self.img_size*self.batch_size,
                              out_features=512)
        self.enc2 = nn.Linear(in_features=512,
                              out_features=self.features*2)
        
        # Decoder
        self.dec1 = nn.Linear(in_features=self.features,
                              out_features=512)
        self.dec2 = nn.Linear(in_features=512,
                              out_features=self.img_size*self.img_size*self.batch_size)
        
    def reparam(self, mu, log_var):
        
        std = torch.exp(0.5*log_var)
        eps = torch.rand_like(std)
        
        return mu + (eps * std)
    
    def forward(self, x):
        
        # Encoding
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, )
        
        mu = x[:, 0, :]
        log_var = x[:, 1, :]
        
        z = self.reparam(mu, log_var)
        
        # Decoding
        x = F.relu(self.dec1(z))
        
        out = torch.sigmoid(self.dec2(x))
        
        return out, mu, log_var
    
    def fLoss(loss, mu, log_var):
        
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return loss + kl
    
    def trainModel(self, metrics, optimizer, dataset, epochs):
        
        self.train()
        
        currLoss = 0.0
        
        train_set = tqdm(dataset)
        
        for i, data in enumerate(train_set):
            
            
            inputs = data.to('cuda') if torch.cuda.is_available() else 'cpu'
            optimizer.zero_grad()
            
            outputs, mu, log_var = self.forward(inputs.view(1, -1))
            
            loss = metrics(outputs, inputs)
            
            loss = self.fLoss(loss, mu, log_var)
            
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
                    raise NotImplementedError
                    # TODO: Output code here
                else:
                    preds[i], currLoss = self._output(metrics, data)
                    
                    loss += currLoss
                val_set.set_description("Validation")
                val_set.set_postfix(loss=loss)
        
        return preds, loss
    
    def _output(self, metrics, data):
        
        x_tensor = data.to('cuda') if torch.cuda.is_available() else 'cpu'
        pred, mu, log_var = self.forward(x_tensor)
        loss = metrics(pred, x_tensor)
        loss = self.fLoss(mu, log_var)
        pred = pred.squeeze().cpu().numpy().round()
        
        return pred.T, loss.item()


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
                        preds[idx], currLoss = self._output(metrics, data[j].unsqueeze(0))
                        idx += 1
                        
                        loss += currLoss
                else:
                    preds[i], currLoss = self._output(metrics, data)
                    
                    loss += currLoss
                val_set.set_description("Validation")
                val_set.set_postfix(loss=loss)
        
        return preds, loss
    
    def _output(self, metrics, data):
        
        x_tensor = data.to('cuda') if torch.cuda.is_available() else 'cpu'
        pred = self.layers(x_tensor)
        loss = metrics(pred, x_tensor)
        pred = pred.squeeze().cpu().numpy().round()
        
        return pred.T, loss.item()

class C_RAE(nn.Module):
    '''
    Convolutional ResNet Autoencoder
    
    Reference:
        CHATHURIKA S. WICKRAMASINGHE , DANIEL L. MARINO ,
        MILOS MANIC - ResNet Autoencoders for Unsupervised Feature
                      Learning From High-Dimensional Data: Deep
                      Models Resistant to Performance Degradation
    
    DOI: 10.1109/ACCESS.2021.3064819
    '''
    def __init__(self):
        super(C_RAE, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=16,
                      kernel_size=3,
                      padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.LeakyReLU(),
            )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=3,
                      padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.LeakyReLU()
            )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=(2, 2)),
            nn.ConvTranspose2d(in_channels=16,
                               out_channels=16),
            nn.LeakyReLU()
            )
        
        self.dec2 = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=(2, 2)),
            nn.ConvTranspose2d(in_channels=16,
                               out_channels=3),
            nn.LeakyReLU()
            )
    
    def forward(self, x):
        
        # Encoder
        x = F.relu(self.enc(x))
        # TODO: Finish architecture