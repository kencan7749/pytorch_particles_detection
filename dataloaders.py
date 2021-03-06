import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms


root_dir = './dataset_sample'

class CloudStorDataset(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, file_list, feat='dual', transform=None, phase='train'):
        self.file_list= file_list
        self.transform = transform
        self.phase=phase

        feat_dict ={
            'dual': [0,4,6,10],
            'single': [0,4],
            'sigle_geometry': [0], #need closely check
            'single_intensity': [4],
        }
        self.feat_dim = feat_dict[feat]
        self.labels_dim = [12, 13, 14]

    def __len__(self): 
        return len(self.file_list)

    def __getitem__(self, index):
        #load
        npy_path = self.file_list[index]
        npy = np.load(npy_path)

        feat=npy[...,self.feat_dim]
        label = npy[...,self.labels_dim]

        if self.transform is not None:
            feat, label = self.transform(feat, label, self.phase)

        return feat, label





if __name__ == '__main__':

    root_dir = './dataset_sample/'

    data_path= glob.glob(root_dir+'/*.npy')

    dset = CloudStorDataset(data_path)

    index = 0
    print(dset.__getitem__(index)[0].shape)


    batch_size = 32
    data_loader = torch.utils.data.DataLoader(
        dset, batch_size=batch_size, shuffle=True
    )

    batch_iterator = iter(data_loader)

    inputs, labels = next(batch_iterator)

    print(inputs.size)
    print(labels.size)