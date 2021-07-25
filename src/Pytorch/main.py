# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 22:57:43 2021

@author: iason
"""

import torch
from Utils import Dataset, predRecon
from Undercomplete import UAutoencoder
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
from Conv_Autoenc import ConvAutoencoder

def main():
    
    device = torch.device('cuda')
    EPOCH=10
    lr = 1e-2
    algo = 'adam'
    img_size = (256, 256)
    segm_size = 4
    batch_size = 1
    
    path = '../dataset/dataset1/'
    
    train, test = train_test_split(os.listdir(path),
                                   train_size=0.6,
                                   shuffle=False)
    
    dataset = Dataset(path=path,
                      img_size=img_size,
                      segm_size=segm_size,
                      batch_size=batch_size,
                      inputs=train[:len(train)//8])
    
    train_set = torch.utils.data.DataLoader(dataset,
                                            shuffle=True)
    
    dataset_test = Dataset(path=path,
                           img_size=img_size,
                           segm_size=segm_size,
                           batch_size=batch_size,
                           inputs=test[:len(test)//8])
    
    test_set = torch.utils.data.DataLoader(dataset_test,
                                           shuffle=True)
    
    model = UAutoencoder(img_size[0]).to(device)
    metrics = nn.MSELoss()
    
    
    weight_decay = 0.00005
    
    learners = {'adam': torch.optim.Adam(params=model.parameters(),
                                         lr=lr,
                                         weight_decay=weight_decay),
                
                'adadelta': torch.optim.Adadelta(params=model.parameters(),
                                                 lr=lr,
                                                 weight_decay=weight_decay),
                
                'adagrad' : torch.optim.Adagrad(params=model.parameters(),
                                                lr=lr,
                                                weight_decay=weight_decay),
                
                'adamw' : torch.optim.AdamW(params=model.parameters(),
                                            weight_decay=weight_decay,
                                            amsgrad=False),
                
                'sparseadam' : torch.optim.SparseAdam(params=model.parameters(),
                                                      lr=lr),
                
                'adamax' : torch.optim.Adamax(params=model.parameters(),
                                              lr=lr,
                                              weight_decay=weight_decay),
                
                'asgd' : torch.optim.ASGD(params=model.parameters(),
                                          lr=lr,
                                          weight_decay=weight_decay),
                
                'lbfgs' : torch.optim.LBFGS(params=model.parameters(),
                                            lr=lr,
                                            max_iter=20),
                
                'rmsprop' : torch.optim.RMSprop(params=model.parameters(),
                                                lr=lr,
                                                weight_decay=weight_decay,
                                                centered=False),
                
                'rprop' : torch.optim.Rprop(params=model.parameters(),
                                            lr=lr),
                
                'sgd' : torch.optim.SGD(params=model.parameters(),
                                        lr=lr,
                                        weight_decay=weight_decay,
                                        nesterov=False),
                }
    
    optimizer = learners[algo]
    
    
    max_score = 0
    lrs = []
    
    for epoch in range(EPOCH):
        
        currLoss = 0.0
        
        
        # print('\nEpoch: {}\n'.format(epoch))
        
        train_set = tqdm(train_set)
        
        for i, data in enumerate(train_set):
            
            inputs = data.to(device)
            
            inputs = inputs.reshape(1, -1)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            loss = metrics(outputs, inputs)
            
            loss.backward()
            
            optimizer.step()
            
            currLoss += loss.item()
            
            train_set.set_description(f"Epoch [{epoch+1}/{EPOCH}]")
            train_set.set_postfix(loss=loss.item())
        
        lrs.append(currLoss / len(train_set))
        
        
        # do something (save model, change lr, etc.)
        if max_score > currLoss / len(train_set):
            max_score = currLoss / len(train_set)
            torch.save(model, './pytorch_best_model.pth')
            print('Model saved!')
            
    return predRecon(test_set)

if __name__ == "__main__":
    preds = main()