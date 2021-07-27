# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 23:03:22 2021

@author: iason
"""

import torch
import os
from PIL import Image
import torchvision.transforms as T

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, inputs, img_size, segm_size=4):
        self.path = path
        self.img_size = img_size
        self.transforms = T.Compose([
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],),
                        ])
        self.segm_size=segm_size
        # self.kernel_size=(self.img_size[0]//(segm_size//2),
        #                   self.img_size[1]//(segm_size//2))
        self.kernel_idx = 0
        
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(inputs))
    
    def __getitem__(self, idx):
        # load images
        img_path = os.path.join(self.path, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        # Return top left kernel
        # if self.kernel_idx == 0:
        #     img = img.crop((0, 0, self.kernel_size[0], self.kernel_size[1]))
        #     # Increment kernel index flag
        #     self.kernel_idx += 1
        # # Return top right kernel
        # elif self.kernel_idx == 1:
        #     img=img.crop((self.kernel_size[0], 0,
        #                      self.img_size[1], self.kernel_size[1]))
        #     # Increment kernel index flag
        #     self.kernel_idx += 1
        # # Return bottom left kernel
        # elif self.kernel_idx == 2:
        #     img=img.crop((0, self.kernel_size[1], self.kernel_size[0],
        #                      self.img_size[1]))
        #     # Increment kernel index flag
        #     self.kernel_idx += 1
        # # Return bootm right kernel
        # else:
        #     img=img.crop((self.kernel_size[0], self.kernel_size[1],
        #                      self.img_size[0], self.img_size[1]))
        #     # Reset kernel index flag
        #     self.kernel_idx = 0
        
        img = self.transforms(img)
        
        return img
    
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