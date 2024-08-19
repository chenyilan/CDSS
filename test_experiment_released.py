# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       test_abdomen
   Project Name:    CDSS
   Author :         Hengrong LAN
   Date:            2022/9/30
   Device:          RTX 3090
-------------------------------------------------
   Change Activity:
                   2024/7/30:
-------------------------------------------------
"""



import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils.trick import *
import torch
import scipy.io as sio
from utils.mydataset import ReconDataset_test
from utils.MeasurementMatrix import Load_Aa_Simulation_PSF
from utils.SubSampling import Creat_Subsampling
import time
import torch.nn as nn
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


checkpoint0 = torch.load('./checkpoint/CDSS_ifUnet_50.ckpt')
model0 = checkpoint0['model']
device = torch.device('cuda:0')
model0 = nn.DataParallel(model0)
model0 = model0.to(device)


batch_size=1
# dataset_pathr='E:/unsupervised_re/data/Ma_data/'
dataset_pathr='./dataset/data/'
test_dataset = ReconDataset_test(dataset_pathr,train=False)
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False)
Matrix_Path='./Matrix/H_128000_128'
# Matrix_Path='./Matrix/H_128000_128_Ma'
Aa = Load_Aa_Simulation_PSF(Matrix_Path)
Aa_pinv = Aa.t()

im_size=128

#source activate pytorch
batch_idx=0

with torch.no_grad():
    for rawdata, out_rawdata, filename in tqdm((test_loader)):
        #rawdata =50 *rawdata.to(device)
        rawdata = 50 * rawdata.to(device)
        # split_mask=Creat_Subsampling(1000,128,0.875).to(device)#
        split_mask = torch.zeros(1000, 128)  # Creat_Subsampling(1000,128,0.5).to(device)#
        split_mask[:, 0:-1:2] = 1
        split_mask = split_mask.to(device)
        #
        data_s1 = torch.mul(rawdata, split_mask)

        b, n, m = rawdata.size()
        data_s1 = data_s1.reshape(b, n * m)  # .permute(0,2,1)
        x1 = torch.sparse.FloatTensor.mm(Aa_pinv, data_s1.t()).t().reshape(-1,1,im_size, im_size).transpose(2, 3)
        end = time.time()
        outputs  = model0(x1)
        t1 = time.time() - end

        outputs_cpu = outputs.to('cpu')
        x1_cpu = x1.to('cpu')
        rawdata_cpu = rawdata.to('cpu')


        raw_data = rawdata_cpu.detach().squeeze().numpy()
        output_data = outputs_cpu.detach().squeeze().numpy()
        bf_img = x1_cpu.detach().squeeze().numpy()
        p_re = out_rawdata.detach().squeeze().numpy()


        batch_idx=batch_idx+1
        # sio.savemat('./test_results/IFUnet/50/' + filename[0],
        #             {'output_data': output_data, 'p_re': p_re, 'bf_img': bf_img, 'raw_data':raw_data})
        



