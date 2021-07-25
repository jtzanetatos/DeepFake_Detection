# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 23:03:22 2021

@author: iason
"""

import torch
import os
import numpy as np
import cv2 as cv
from PIL import Image
import torchvision.transforms as T

# TODO: Remove masks (?)

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
        self.kernel_size=(self.img_size[0]//(segm_size//2),
                          self.img_size[1]//(segm_size//2))
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
        
        img = self.transforms(img).unsqueeze(0)
        
        return img
    
    def __len__(self):
        return len(self.imgs)

def predRecon(test_dataset):
    
    
    best_model = torch.load('./pytorch_best_model.pth')
    
    preds = np.zeros(len(test_dataset), dtype=object)
    
    with torch.no_grad():
        for i, data in enumerate(test_dataset):
            
            image = data.cpu().numpy()[0, 0, :,:,:].T
            
            x_tensor = data.to('cuda').unsqueeze(0)
            pr_mask = best_model(x_tensor)
            pr_mask = pr_mask.squeeze().cpu().numpy().round()
            
            # Cast to uint8
            pr_mask = np.uint8(pr_mask.reshape(image.shape)*255)
            
            preds[i] = pr_mask
    
    return preds