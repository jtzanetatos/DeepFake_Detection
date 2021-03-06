# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 22:57:43 2021

@author: iason
"""

import torch
from Utils import Dataset, optimizers, loadDataset
from Models import (UAutoencoder, ConvAutoencoder, SparceAutoencoder,
                    C_RAE, ConvVAE)
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def main():
    
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}
    EPOCH=15
    lr = 1e-3
    alg = 'adam'
    img_size = (256, 256)
    segm_size = 1
    batch_size = 4
    
    path = '../dataset/dataset2/'
    
    # TODO: Dictionary
    train_set, train_paths, val_set, val_path, test_set, test_path, test_labels = loadDataset(path)
    
    train, val = train_test_split(os.listdir(path),
                                   train_size=0.6,
                                   shuffle=False)
    
    dataset = Dataset(path=train_paths,
                      img_size=img_size,
                        inputs=train_set) #[:len(train)//8]
                        # inputs=train[:len(train)//4])
    
    train_set = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            **kwargs)
    
    dataset_val = Dataset(path=val_path,
                            img_size=img_size,
                            inputs=val_set)
                            # inputs=val[:len(val)//4])
    
    val_set = torch.utils.data.DataLoader(dataset_val,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           **kwargs)
    
    dataset_test = Dataset(path=test_path,
                       img_size=img_size,
                       inputs=test_set)
    
    test_set = torch.utils.data.DataLoader(dataset_test,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           **kwargs)
    # OOM for Undercomplete encoder with batch size > 1
    # model = UAutoencoder(img_size[0], batch_size).to(device)
    # model = SparceAutoencoder(img_size[0], batch_size, sparcity='l1').to(device)
    # model = ConvAutoencoder(batch_size).to(device)
    model = C_RAE(img_size[0], batch_size).to(device)
    # model = ConvVAE(batch_size).to(device)
    
    metrics=nn.MSELoss()
    
    weight_decay = 0.0001
    
    optimizer = optimizers(alg, lr, weight_decay, model)
    
    
    # max_score = 0.0
    train_loss = []
    val_loss = []
    
    # Load pretrained weights
    model.loadWeights()
    
    for epoch in range(EPOCH):
        
        # train_epoch_loss = model.trainModel(metrics, optimizer, train_set, EPOCH)
        # preds, val_epoch_loss = model.evaluate(metrics, val_set, img_shape=(256, 256, 3))
        
        train_epoch_loss = model.trainClass(metrics, optimizer, train_set, EPOCH)
        preds, val_epoch_loss = model.evaluateClass(metrics, val_set, img_shape=(256, 256, 3))
        
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        
        # do something (save model, change lr, etc.)
        # if max_score < val_epoch_loss:
            # max_score = val_epoch_loss
            # model.saveWeights()
    # Save weights
    # model.saveWeights()
    preds, test_loss = model.evaluateClass(metrics, test_set, img_shape=(256, 256, 3))
    # Clear gpu memory
    torch.cuda.empty_cache()
    return EPOCH, train_loss, val_loss, preds, test_loss, test_labels

if __name__ == "__main__":
    # try:
    epochs, train_loss, val_loss, preds, test_loss, test_labels = main()
    # except Exception as e:
    #     # Clear gpu memory
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    #     sys.exit(print(e))
    # TODO: Visualizations of metrics, results
    epochs = np.arange(1, epochs+1)
    plt.plot(epochs, train_loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()