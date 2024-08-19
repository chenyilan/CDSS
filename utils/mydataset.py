# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       dataset
   Author :         Hengrong LAN
   Date:            2018/12/26
-------------------------------------------------
   Change Activity:
                   2018/12/26:
-------------------------------------------------
"""

import numpy as np
import torch
import scipy
import os
import os.path
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
import scipy.io as scio


def np_range_norm(image, maxminnormal=True, range1=True):

    if image.ndim == 2 or (image.ndim == 3 and image.shape[0] == 1):
        if maxminnormal:
            _min = image.min()
            _range = image.max() - image.min()
            narmal_image = (image - _min) / _range
            if range1:
               narmal_image = (narmal_image - 0.5) * 2
        else:
            _mean = image.mean()
            _std = image.std()
            narmal_image = (image - _mean) / _std

    return narmal_image



class ReconDataset(data.Dataset):
    __inputdata = []
    __outputimg = []
 

    def __init__(self,root, train=True,transform=None):
        self.__inputdata = []
        self.__outputdata = []
        self.__outputimg = []

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        if train:
            folder = root + "Train/"
        else:
            folder = root + "Test/"
            
        
        
        for file in os.listdir(folder):
            #print(file)
            matdata = scio.loadmat(folder + file)
            raw_data = -1*matdata['Sinogram']
            prt = matdata['p_re']

            #for index in range(0,nviews):
            #    prt[index,:,:]=np_range_norm(prt[index,:,:], maxminnormal=True, range1=False)
            self.__inputdata.append(raw_data[1000:2001,0:-1:4])

            
            self.__outputimg.append(prt[np.newaxis,:,:])




        
            


    def __getitem__(self, index):

        rawdata =  self.__inputdata[index] 
        DAS = self.__outputimg[index]
           

        rawdata = torch.Tensor(rawdata)
        DAS = torch.Tensor(DAS)

        return rawdata, DAS

    def __len__(self):
        return len(self.__inputdata)


class ReconDataset_test(data.Dataset):
    __inputdata = []
    __outputimg = []

    def __init__(self, root, train=True, transform=None):
        self.__inputdata = []
        self.__outputdata = []
        self.__outputimg = []
        self.__filename = []
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        if train:
            folder = root + "Train/"
        else:
            folder = root + "Test/"

        for file in os.listdir(folder):
            # print(file)
            matdata = scio.loadmat(folder + file)
            raw_data = -1 * matdata['Sinogram']
            prt = matdata['p_re']
            file_name = file


            # for index in range(0,nviews):
            #    prt[index,:,:]=np_range_norm(prt[index,:,:], maxminnormal=True, range1=False)
            self.__inputdata.append(raw_data[1000:2001, 0:-1:4])

            self.__outputimg.append(prt[np.newaxis, :, :])
            self.__filename.append(file_name)

    def __getitem__(self, index):

        rawdata = self.__inputdata[index]
        DAS = self.__outputimg[index]
        name = self.__filename[index]

        rawdata = torch.Tensor(rawdata)
        DAS = torch.Tensor(DAS)

        return rawdata, DAS, name

    def __len__(self):
        return len(self.__inputdata)



if __name__ == "__main__":
    dataset_pathr = '../../data/mice_data/'

    mydataset = ReconDataset(dataset_pathr,train=False)
    #print(mydataset.__getitem__(3))
    train_loader = DataLoader(
        mydataset,
        batch_size=1, shuffle=True)
    batch_idx, (rawdata, DAS) = list(enumerate(train_loader))[0]

    print(rawdata.size())
    print(rawdata.max())
    print(rawdata.min())
    print(mydataset.__len__())






