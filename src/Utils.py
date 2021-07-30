# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 23:03:22 2021

@author: iason
"""

import torch
import os
from PIL import Image
import torchvision.transforms as T
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, inputs, img_size):
        self.path = path
        self.img_size = img_size
        
        self.transforms = T.Compose([
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],),
                        ])
        
        
        # load all image files, sorting them to
        # ensure that they are aligned
        # self.imgs = list(sorted(inputs))
        self.imgs = inputs
    
    def __getitem__(self, idx):
        
        # if isinstance(self.path, np.ndarray):
        img_path = os.path.join(self.path[idx], self.imgs[idx])
        
        # else:
            # load images
        # img_path = os.path.join(self.path, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        
        # if isinstance(self.path, np.ndarray):
            # Get label from path
        if 'real' in self.path[idx]:
            y = 1
        else:
            y = 0
        
        return self.transforms(img), torch.tensor(y).float()
        
        # else:
        # img = self.transforms(img)
        
        # return img
    
    def __len__(self):
        return len(self.imgs)

def optimizers(alg, lr, weight_decay, model):
    
    
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
    
    return learners[alg]

def loadDataset(path):
    
    # Real images directory
    real = os.listdir(os.path.join(path, 'real'))
    
    # Fake images directory
    fake = os.listdir(os.path.join(path, 'fake'))
    
    train_set = real[:400] + fake[400:800]
    train_paths = np.zeros(800, dtype=object)
    train_paths[:400] = os.path.join(path, 'real')
    train_paths[400:] = os.path.join(path, 'fake')
    
    
    val_set = real[800:850] + fake[800:850]
    
    val_path = np.zeros(len(val_set), dtype=object)
    val_path[:50] = os.path.join(path, 'real')
    val_path[50:] = os.path.join(path, 'fake')
    
    test_set = real[850:900] + fake[850:900]
    
    test_path = np.zeros(len(test_set), dtype=object)
    test_path[:50] = os.path.join(path, 'real')
    test_path[50:] = os.path.join(path, 'fake')
    test_labels = torch.zeros(len(test_set))
    test_labels[:50] = 1
    
    return (train_set, train_paths, val_set,
            val_path, test_set, test_path, test_labels)
