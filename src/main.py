# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 22:57:43 2021

@author: iason
"""

import torch
from Utils import Dataset, optimizers
from Models import UAutoencoder, ConvAutoencoder
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys

def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(x)
        # Computes loss
        loss = loss_fn(x, yhat)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()
    
    # Returns the function that will be called inside the train loop
    return train_step

def main():
    
    
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}
    EPOCH=4
    lr = 1e-3
    alg = 'adam'
    img_size = (256, 256)
    segm_size = 1
    batch_size = 1
    
    path = '../dataset/dataset1/'
    
    train, test = train_test_split(os.listdir(path),
                                   train_size=0.6,
                                   shuffle=False)
    
    dataset = Dataset(path=path,
                      img_size=img_size,
                      segm_size=segm_size,
                      inputs=train[:len(train)//8])
    
    train_set = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            **kwargs)
    
    dataset_test = Dataset(path=path,
                           img_size=img_size,
                           segm_size=segm_size,
                           inputs=test[:len(test)//8])
    
    test_set = torch.utils.data.DataLoader(dataset_test,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           **kwargs)
    # OOM for Undercomplete encoder with batch size > 1
    model = UAutoencoder(img_size[0], batch_size=1).to(device)
    # model = ConvAutoencoder().to(device)
    metrics = nn.BCELoss()
    
    weight_decay = 0.00005
    
    optimizer = optimizers(alg, lr, weight_decay, model)
    
    
    max_score = 0.0
    lrs = []
    
    # train_step = make_train_step(model, metrics, optimizer)
    
    for epoch in range(EPOCH):
        
        currLoss = 0.0
        
        
        # print('\nEpoch: {}\n'.format(epoch))
        
        train_set = tqdm(train_set)
        
        for i, data in enumerate(train_set):
            
            inputs = data.to(device)
            
            if model.__class__.__name__=='UAutoencoder':
                inputs = inputs.view(-1, img_size[0]*img_size[1])
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            loss = metrics(outputs, inputs)
            
            loss.backward()
            
            optimizer.step()
            
            currLoss += loss.item()
            # loss = train_step(data.to(device))
            # lrs.append(loss)
            
            train_set.set_description(f"Epoch [{epoch+1}/{EPOCH}]")
            train_set.set_postfix(loss=loss.item())
        
        lrs.append(currLoss / len(train_set))
        
        
        # do something (save model, change lr, etc.)
        if max_score < currLoss / len(train_set):
            max_score = currLoss / len(train_set)
            torch.save(model, './pytorch_best_model.pth')
            print('Model saved!')
    # Clear gpu memory
    torch.cuda.empty_cache()
    
    return model.predict(model, test_set, batch_size, img_shape=(256, 256, 3)), lrs

if __name__ == "__main__":
    try:
        preds, loss = main()
    except Exception as e:
        # Clear gpu memory
        torch.cuda.empty_cache()
        sys.exit(print(e))