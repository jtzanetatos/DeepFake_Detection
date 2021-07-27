# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 22:57:43 2021

@author: iason
"""

import torch
from Utils import Dataset, optimizers
from Models import UAutoencoder, ConvAutoencoder, SparceAutoencoder, VarAutoencoder
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys

def main():
    
    
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}
    EPOCH=1
    lr = 1e-3
    alg = 'adam'
    img_size = (256, 256)
    segm_size = 1
    batch_size = 2
    
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
    
    dataset_val = Dataset(path=path,
                           img_size=img_size,
                           segm_size=segm_size,
                           inputs=test[:len(test)//8])
    
    val_set = torch.utils.data.DataLoader(dataset_val,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           **kwargs)
    # OOM for Undercomplete encoder with batch size > 1
    model = UAutoencoder(img_size[0], batch_size).to(device)
    # model = SparceAutoencoder(img_size[0], batch_size, sparcity='kl').to(device)
    # model = ConvAutoencoder(batch_size).to(device)
    # model = VarAutoencoder(img_size[0], batch_size).to(device)
    metrics = nn.BCELoss()
    
    weight_decay = 0.00005
    
    optimizer = optimizers(alg, lr, weight_decay, model)
    
    
    max_score = 0.0
    train_loss = []
    val_loss = []
    
    
    for epoch in range(EPOCH):
        
        train_epoch_loss = model.trainModel(metrics, optimizer, train_set, EPOCH)
        _, val_epoch_loss = model.evaluate(metrics, val_set, img_shape=(256, 256, 3))
        
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        
        # do something (save model, change lr, etc.)
        if max_score < train_epoch_loss:
            max_score = train_epoch_loss
            torch.save(model, './pytorch_best_model.pth')
            print('Model saved!')
    
    preds, _ = model.evaluate(metrics, val_set, img_shape=(256, 256, 3))
    # Clear gpu memory
    torch.cuda.empty_cache()
    # return model.predict(metrics, test_set, batch_size, img_shape=(256, 256, 3)), lrs
    return train_loss, val_loss, preds

if __name__ == "__main__":
    # try:
    train_loss, val_loss, preds = main()
    # except Exception as e:
    #     # Clear gpu memory
    #     torch.cuda.empty_cache() if torch.cuda.is_available() else pass
    #     sys.exit(print(e))
    # TODO: Visualizations of metrics, results