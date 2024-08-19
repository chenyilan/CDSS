import torch
import numpy as np
from torch.autograd import Variable


def Creat_Subsampling(data_len,channel_number,sub_rate):
    sample_num = np.fix(torch.tensor(sub_rate * channel_number)).astype(int)
    subsample_mask = torch.ones(data_len,channel_number)
    sample = torch.randperm(channel_number)
    sample = sample[:sample_num]
    subsample_mask[:,sample]=0
    #sample = sample[0:sample_num]
    #Subsampled_matrix = torch.eye(channel_number)
    #Subsampled_matrix[sample,sample]=0
    #Subsampled_matrix=Subsampled_matrix.view(1,channel_number,channel_number)

    return subsample_mask


def main():
    # x = Variable(torch.FloatTensor([[[1,2],[2,3]],[[1,2],[2,3]]]).view(1,2,2,2), requires_grad=True)
    # x = Variable(torch.FloatTensor([[[3,1],[4,3]],[[3,1],[4,3]]]).view(1,2,2,2), requires_grad=True)
    # x = Variable(torch.FloatTensor([[[1,1,1], [2,2,2],[3,3,3]],[[1,1,1], [2,2,2],[3,3,3]]]).view(1, 2, 3, 3), requires_grad=True)
    x = torch.FloatTensor(np.random.random((2,1000, 128)))
    print(x.shape)
    mask=Creat_Subsampling(1000,128,0.5)
    print(mask)
    print(mask.shape)
    z = torch.mul(x,mask)
    y=x-z
    print (x)
    print (z)
    print(z.shape)
    print(y)



if __name__ == '__main__':
    main()
