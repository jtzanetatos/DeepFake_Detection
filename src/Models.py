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
        
        self.classifier = nn.Sequential(
                            nn.Linear(self.inShape*self.inShape*self.batch_size * 7 * 7,
                                      4096),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(4096, 4096),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(4096, 1)
                            )
        self.avgpool = nn.AdaptiveAvgPool2d(7)
    
    def forward(self, x):
        
        return self.layers(x)
    
    def classForward(self, x):
        
        x = self.avgpool(x)
        return self.classifier(x.view(1, -1))
    
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
    
    def trainClassifier(self, metrics, dataset, img_shape):
        pass
    
    def saveWeights(self):
        torch.save(self.state_dict(), './pytorch_'+self.__class__.__name__+'_weights.pth')
        print("Weights saved successfuly.")
    
    def loadWeights(self):
        self.load_state_dict(torch.load('./pytorch_'+self.__class__.__name__+'_weights.pth'))
        print("Weights loaded successfuly.")


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
    
    def saveWeights(self):
        torch.save(self.state_dict(), './pytorch_'+self.__class__.__name__+self.sparcity.lower()+'_weights.pth')
        print("Weights saved successfuly.")
    
    def loadWeights(self):
        self.load_state_dict(torch.load('./pytorch_'+self.__class__.__name__+self.sparcity.lower()+'_weights.pth'))
        print("Weights loaded successfuly.")

# BUG: Loss function produces error, or nan values
# class VarAutoencoder(nn.Module):
    
#     def __init__(self, img_size, batch_size, features=32):
#         super(VarAutoencoder, self).__init__()
        
#         self.img_size = img_size
#         self.batch_size = batch_size
#         self.features = features
#         self.epochIdx = 1
        
#         # Encoder
#         self.enc1 = nn.Linear(in_features=self.img_size*self.img_size*self.batch_size,
#                               out_features=512)
#         self.enc2 = nn.Linear(in_features=512,
#                               out_features=self.features*2)
        
#         # Decoder
#         self.dec1 = nn.Linear(in_features=self.features,
#                               out_features=512)
#         self.dec2 = nn.Linear(in_features=512,
#                               out_features=self.img_size*self.img_size*self.batch_size)
        
#     def _reparam(self, mu, log_var):
        
#         std = torch.exp(0.5*log_var)
#         eps = torch.randn_like(std)
        
#         return mu + (eps * std)
    
#     def forward(self, x):
        
#         # Encoding
#         x = F.relu(self.enc1(x))
#         x = self.enc2(x).view(-1, 2, self.features)
        
#         mu = x[:, 0, :]
#         log_var = x[:, 1, :]
        
#         z = self._reparam(mu, log_var)
        
#         # Decoding
#         x = F.relu(self.dec1(z))
        
#         out = torch.sigmoid(self.dec2(x))
        
#         return out, mu, log_var
    
#     def _fLoss(self, loss, mu, log_var):
        
#         kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
#         return loss + kl
    
#     def trainModel(self, metrics, optimizer, dataset, epochs):
        
#         self.train()
        
#         currLoss = 0.0
        
#         train_set = tqdm(dataset)
        
#         for i, data in enumerate(train_set):
            
#             for k in range(data.shape[1]):
#                 inputs = data[0, k].to('cuda') if torch.cuda.is_available() else 'cpu'
#                 optimizer.zero_grad()
                
#                 outputs, mu, log_var = self.forward(inputs.view(1, -1))
                
#                 loss = metrics(outputs, inputs.view(1, -1))
                
#                 loss = self._fLoss(loss, mu, log_var)
                
#                 currLoss += loss.item()
                
#                 loss.backward()
                
#                 optimizer.step()
                
#             train_set.set_description(f"Epoch [{self.epochIdx+1}/{epochs}]")
#             train_set.set_postfix(loss=loss.item())
            
#         self.epochIdx += 1
        
#         return currLoss / len(dataset)
    
#     def evaluate(self, metrics, dataset, img_shape):
        
#         self.eval()
        
#         preds = np.zeros(len(dataset)*self.batch_size, dtype=object)
        
#         idx = 0
        
#         loss = 0.0
        
#         val_set = tqdm(dataset)
        
#         with torch.no_grad():
#             for i, data in enumerate(val_set):
                
#                 if data.shape[0] > 1:
#                     raise NotImplementedError
#                     # TODO: Output code here
#                 else:
#                     tempPred = np.zeros((img_shape[0]*img_shape[1], img_shape[2]))
#                     for k in range(img_shape[2]):
#                         tempPred[:,k], currLoss = self._output(metrics, data[0, k].view(1, -1))
#                         loss += currLoss
#                     preds[idx] = tempPred.reshape(img_shape)
#                     idx += 1
#                 val_set.set_description("Validation")
#                 val_set.set_postfix(loss=loss)
        
#         return preds, loss
    
#     def _output(self, metrics, data):
        
#         x_tensor = data.to('cuda') if torch.cuda.is_available() else 'cpu'
#         pred, mu, log_var = self.forward(x_tensor)
#         loss = metrics(pred, x_tensor)
#         loss = self._fLoss(loss, mu, log_var)
#         pred = pred.squeeze().cpu().numpy().round()
        
#         return pred.T, loss.item()
    
#     def saveWeights(self):
#         torch.save(self.state_dict(), './pytorch_'+self.__class__.__name__+'_weights.pth')
    
#     def loadWeights(self):
#         self.load_state_dict(torch.load('./pytorch_'+self.__class__.__name__+'_weights.pth'))

class ConvVAE(nn.Module):
    
    def __init__(self, batch_size, init_filt=32, features=100):
        super(ConvVAE, self).__init__()
        
        self.batch_size = batch_size
        self.init_filt = init_filt
        self.features = features
        self.epochIdx = 0
        
        # Encoder
        self.enc = nn.Sequential(nn.Conv2d(in_channels=3,
                                           out_channels=self.init_filt,
                                           kernel_size=4,
                                           stride=2,
                                           padding=2),
                                 nn.ReLU(),
                                 
                                 nn.Conv2d(in_channels=self.init_filt,
                                           out_channels=self.init_filt*2,
                                           kernel_size=4,
                                           stride=2,
                                           padding=2),
                                 
                                 nn.ReLU(),
                                 
                                 nn.Conv2d(in_channels=self.init_filt*2,
                                           out_channels=self.init_filt*4,
                                           kernel_size=4,
                                           stride=2,
                                           padding=2),
                                 
                                 nn.ReLU(),
                                 
                                 nn.Conv2d(in_channels=self.init_filt*4,
                                           out_channels=self.init_filt*8,
                                           kernel_size=4,
                                           stride=2,
                                           padding=2),
                                 
                                 nn.ReLU(),
                                 
                                 nn.Conv2d(in_channels=self.init_filt*8,
                                           out_channels=self.init_filt*16,
                                           kernel_size=4,
                                           stride=2,
                                           padding=2),
                                 nn.ReLU()
                                 )
        
        
        # Decoder
        self.dec = nn.Sequential(nn.ConvTranspose2d(in_channels=self.init_filt*16,
                                                    out_channels=self.init_filt*8,
                                                    kernel_size=4,
                                                    stride=2,
                                                    padding=0),
                                 nn.ReLU(),
                                 
                                 nn.ConvTranspose2d(in_channels=self.init_filt*8,
                                                    out_channels=self.init_filt*4,
                                                    kernel_size=4,
                                                    stride=2,
                                                    padding=0),
                                 
                                 nn.ReLU(),
                                 
                                 nn.ConvTranspose2d(in_channels=self.init_filt*4,
                                                    out_channels=self.init_filt*2,
                                                    kernel_size=4,
                                                    stride=2,
                                                    padding=0),
                                 
                                 nn.ReLU(),
                                 
                                 nn.ConvTranspose2d(in_channels=self.init_filt*2,
                                                    out_channels=self.init_filt,
                                                    kernel_size=4,
                                                    stride=3,
                                                    padding=1),
                                 
                                 nn.ReLU(),
                                 
                                 nn.ConvTranspose2d(in_channels=self.init_filt,
                                                    out_channels=3,
                                                    kernel_size=2,
                                                    stride=4,
                                                    padding=1),
                                 
                                 nn.Sigmoid()
                                 )
        self.fc1 = nn.Linear(self.init_filt*16, self.init_filt*16*2)
        self.fc_mu = nn.Linear(self.init_filt*16*2, self.features)
        self.fc_logvar = nn.Linear(self.init_filt*16*2, self.features)
        self.fc2 = nn.Linear(self.features, self.init_filt*16)
        
    def _reparam(self, mu, log_var):
        
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        
        return mu + (eps * std)
    
    def forward(self, x):
        
        # Encoder
        x = self.enc(x)
        
        batch = x.shape[0]
        
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        
        hidden = self.fc1(x)
        
        mu = self.fc_mu(hidden)
        log_var = self.fc_logvar(hidden)
        
        # Latent vector through reparameterization
        z = self._reparam(mu, log_var)
        z = self.fc2(z)
        
        z = z.view(-1, self.init_filt*16, 1, 1)
        
        # Decoder
        x = self.dec(z)
        
        return x, mu, log_var
    
    def _floss(self, loss, mu, log_var):
        
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return loss + kld
    
    def trainModel(self, metrics, optimizer, dataset, epochs):
        
        self.train()
        
        currLoss = 0.0
        
        train_set = tqdm(dataset)
        
        for i, data in enumerate(train_set):
            inputs = data.to('cuda') if torch.cuda.is_available() else 'cpu'
            
            optimizer.zero_grad()
            
            pred, mu, log_var = self.forward(inputs)
            
            loss = metrics(pred, inputs)
            
            loss = self._floss(loss, mu, log_var)
            
            loss.backward()
            
            currLoss += loss.item()
            optimizer.step()
            
            train_set.set_description(f"Epoch [{self.epochIdx+1}/{epochs}]")
            train_set.set_postfix(loss=loss.item())
        self.epochIdx += 1
        
        return currLoss / len(dataset)
    
    def evaluate(self, metrics, dataset, img_shape):
        
        self.eval()
        
        currLoss = 0.0
        
        preds = np.zeros(len(dataset)*self.batch_size, dtype=object)
        
        val_set = tqdm(dataset)
        
        idx = 0
        
        with torch.no_grad():
            for i, data in enumerate(val_set):
                
                if data.shape[0] > 1:
                    for j, batch in enumerate(data):
                        preds[idx], loss = self._output(metrics, data[j].unsqueeze(0))
                        idx += 1
                        
                        currLoss += loss
                else:
                    pred, loss = self._output(metrics, data)
                    
                    preds[i] = pred
                    
                    currLoss += loss
                
                val_set.set_description("Validation")
                val_set.set_postfix(loss=currLoss)
            
            loss = currLoss / (len(dataset)*self.batch_size)
        
        return preds, loss
    
    def _output(self, metrics, data):
        
        x_tensor = data.to('cuda') if torch.cuda.is_available() else 'cpu'
        pred, mu, log_var = self.forward(x_tensor)
        loss = metrics(pred, x_tensor)
        loss = self._floss(loss, mu, log_var)
        pred = pred.squeeze().cpu().numpy().round()
        
        return pred.T, loss.item()
    
    def saveWeights(self):
        torch.save(self.state_dict(), './pytorch_'+self.__class__.__name__+'_weights.pth')
        print("Weights saved successfuly.")
    
    def loadWeights(self):
        self.load_state_dict(torch.load('./pytorch_'+self.__class__.__name__+'_weights.pth'))
        print("Weights loaded successfuly.")

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
            nn.Sigmoid()
            )
        
        self.classifier = nn.Sequential(
                    nn.Linear(256*256*3*self.batch_size,
                              256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                    )
    
    def forward(self, x):
        
        return self.layers(x)
    
    def classForward(self, x):
        
        x = self.forward(x)
        return self.classifier(x.view(1, -1))
    
    def trainModel(self, metrics, optimizer, dataset, epochs):
        
        self.train()
        
        currLoss = 0.0
        
        train_set = tqdm(dataset)
        
        for i, data in enumerate(train_set):
            
            
            inputs = data.to('cuda') if torch.cuda.is_available() else 'cpu'
            optimizer.zero_grad()
            
            outputs = self.forward(inputs)
            
            loss = metrics(outputs, inputs)
            
            loss.backward()
            
            optimizer.step()
            
            currLoss += loss.item()
            
            train_set.set_description(f"Epoch [{self.epochIdx+1}/{epochs}]")
            train_set.set_postfix(loss=loss.item())
            
        self.epochIdx += 1
        
        return currLoss / len(dataset)
    
    def trainClass(self, metrics, optimizer, dataset, epochs):
        
        self.train()
        
        currLoss = 0.0
        
        train_set = tqdm(dataset)
        
        for i, data in enumerate(train_set):
            
            
            inputs = data.to('cuda') if torch.cuda.is_available() else 'cpu'
            optimizer.zero_grad()
            
            outputs = self.classForward(inputs)
            
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
                
                if self.batch_size > 1:
                    for j, batch in enumerate(data):
                        preds[idx], currLoss = self._output(metrics, data[j].unsqueeze(0))
                        idx += 1
                        
                        loss += currLoss
                else:
                    preds[i], currLoss = self._output(metrics, data)
                    
                    loss += currLoss
                val_set.set_description("Validation")
                val_set.set_postfix(loss=loss)
        
        return preds, loss / (len(dataset)*self.batch_size)
    
    
    def _output(self, metrics, data):
        
        x_tensor = data.to('cuda') if torch.cuda.is_available() else 'cpu'
        pred = self.forward(x_tensor)
        loss = metrics(pred, x_tensor)
        pred = pred.squeeze().cpu().numpy().round()
        
        return pred.T, loss.item()
    
    def evaluateClass(self, metrics, dataset, img_shape):
        
        self.eval()
        
        preds = np.zeros(len(dataset)*self.batch_size, dtype=object)
        
        idx = 0
        
        loss = 0.0
        
        val_set = tqdm(dataset)
        
        with torch.no_grad():
            for i, data in enumerate(val_set):
                
                if data.shape[0] > 1:
                    for j, batch in enumerate(data):
                        preds[idx], currLoss = self._outputClass(metrics, data[j].unsqueeze(0))
                        idx += 1
                        
                        loss += currLoss
                else:
                    preds[i], currLoss = self._outputClass(metrics, data)
                    
                    loss += currLoss
                val_set.set_description("Validation")
                val_set.set_postfix(loss=loss)
        
        return preds, loss / (len(dataset)*self.batch_size)
    
    
    def _outputClass(self, metrics, data):
        
        x_tensor = data.to('cuda') if torch.cuda.is_available() else 'cpu'
        pred = self.classForward(x_tensor)
        loss = metrics(pred, x_tensor)
        pred = pred.squeeze().cpu().numpy().round()
        
        return pred.T, loss.item()
    
    def saveWeights(self):
        torch.save(self.state_dict(), './pytorch_'+self.__class__.__name__+'_weights.pth')
        print("Weights saved successfuly.")
    
    def loadWeights(self):
        self.load_state_dict(torch.load('./pytorch_'+self.__class__.__name__+'_weights.pth'))
        print("Weights loaded successfuly.")

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
    def __init__(self, img_size, batch_size):
        super(C_RAE, self).__init__()
        
        self.batch_size = batch_size
        self.epochIdx = 0
        self.inShape = img_size
        
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=2,
                      padding=1,
                      stride=1),
            nn.LeakyReLU(),
            
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=2,
                      padding=0,
                      stride=1),
            
            nn.LeakyReLU()
            )
        
        self.resenc = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=16,
                      kernel_size=2,
                      padding=0,
                      stride=2),
            
            nn.LeakyReLU()
            )
        
        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(in_channels=3,
                               out_channels=3,
                               kernel_size=3,
                               padding=1,
                               stride=1),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(in_channels=3,
                               out_channels=3,
                               kernel_size=3,
                               padding=1,
                               stride=1),
            nn.LeakyReLU()
            )
        
        self.resdec = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16,
                               out_channels=3,
                               kernel_size=2,
                               padding=0,
                               stride=2),
            nn.LeakyReLU()
            )
        
        self.batchNorm1 = nn.BatchNorm2d(num_features=16)
    
    def forward(self, x):
        
        # Encoder
        res = self.resenc(x) # Residual layers
        x = self.enc(res)
        
        # Add residual layers output
        x = torch.add(x, res)
        x = self.batchNorm1(x)
        
        # Decoder
        res = self.resdec(x) # Residual layers
        
        x = self.dec(res)
        
        return torch.sigmoid(torch.add(x, res))
    
    def trainModel(self, metrics, optimizer, dataset, epochs):
        
        self.train()
        
        currLoss = 0.0
        
        train_set = tqdm(dataset)
        
        for i, data in enumerate(train_set):
            
            
            inputs = data.to('cuda') if torch.cuda.is_available() else 'cpu'
            optimizer.zero_grad()
            
            outputs = self.forward(inputs)
            
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
                
                if self.batch_size > 1:
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
        pred = self.forward(x_tensor)
        loss = metrics(pred, x_tensor)
        pred = pred.squeeze().cpu().numpy().round()
        
        return pred.T, loss.item()
    
    def saveWeights(self):
        torch.save(self.state_dict(), './pytorch_'+self.__class__.__name__+'_weights.pth')
        print("Weights saved successfuly.")
    
    def loadWeights(self):
        self.load_state_dict(torch.load('./pytorch_'+self.__class__.__name__+'_weights.pth'))
        print("Weights loaded successfuly.")