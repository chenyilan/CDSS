import time
from .DataInteract import MyMat
import torch
import numpy as np
from scipy import sparse
import scipy.io as scio
from torch.autograd import Variable

def Load_Aa_Simulation_PSF(Path):
    start_time = time.time()
    print('Start loading Measurement Matrix')
    # Measurement = MyMat('/home/zjk/PACT_DL/Simulation/MeasurementMatrix/Indexsparse_Pytorch_Simulation.mat').LoadInMat()
    Measurement = scio.loadmat(Path)
    # Measurement = MyMat('/home/zjk/PACT_DL/Simulation/20210626/Measurement/Indexsparse_Python_Normalize.mat').LoadInMat()
    # Establish the pytorch sparse matrix
    indexsparse = Measurement['indexsparse']
    #indexsparse_2 = Measurement['indexsparse']
    #indexsparse_value = Measurement['indexsparse_value']
    index1 = indexsparse[0,:]
    index2 = indexsparse[1,:]
    indexsparse_value =indexsparse[2,:]
    ##--------------- Original code ---------------##
    i = torch.squeeze(torch.FloatTensor([index1, index2]).long())
    v = torch.squeeze(torch.FloatTensor(indexsparse_value))
    Aa = torch.sparse.FloatTensor(i, v, torch.Size([1000 * 128, 128 * 128])).requires_grad_(False).cuda()
    print('Loading Measurement Matrix cost time: '+ str(time.time()-start_time) + ' seconds')
    return Aa


def Load_Hil(Path):
    Hil = MyMat(Path).LoadInMat()
    Hil = np.asarray(Hil['Hil'])
    return Hil


if __name__ == "__main__":

    #testing
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.FloatTensor(np.random.random((2, 1000, 128))).to(device)
    b,n,m=x.size()
    x = x.permute(0,2,1).contiguous().view(b,n*m)
    #device =  torch.device('cuda:1')
    

    Matrix_Path='../Matrix/H_128000_128'
    Aa = Load_Aa_Simulation_PSF(Matrix_Path)
    print(Aa.shape)
    print(x.shape)

    Proxy = torch.sparse.FloatTensor.mm(Aa.t(), x.t()).t().view(-1, 128, 128)

    print(Proxy.shape)