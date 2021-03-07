import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms


root_dir = './dataset_sample'
metadata_dir = './dataset'
class CloudStorDataset(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, file_list, feat='dual', transform=None, phase='train', metadata_dir='./dataset'):
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

        self.metadata = np.load(os.path.join(metadata_dir, 'metadata.npy'))

    def __len__(self): 
        return len(self.file_list)

    def __getitem__(self, index):
        #load
        npy_path = self.file_list[index]
        npy = np.load(npy_path)
        npy_path = npy_path.replace('\\', '/')
        meta_index = int(npy_path.split('/')[-1][0]) #<= extract "1"-dust ..
        meta = self.metadata[meta_index -1]
        meta_range = meta[:2].astype(np.int64)
        meta_vector = meta[2:]

        feat=npy[...,self.feat_dim]#[:, meta_range[0]: meta_range[1]]
        label = npy[...,self.labels_dim]#[:, meta_range[0]: meta_range[1]]

        if self.transform is not None:
            feat, label = self.transform(feat, label, meta_vector)

        return feat, label


class CloudStorTransformer(object):

    def __init__(self, width=2172, width_pixel=512, probability=0.5):
        self.width=width
        self.width_pixel = width_pixel

        self.flip_probability = probability
        pass
    def __call__(self, x, y, meta_vector):
        ## Take random the snippet of width 512
        # Take random values snippet around polar angle = pi/2 (y-axis) for the two dust datasets of width 512 
        middle_angle = np.random.uniform(meta_vector[0], meta_vector[1])
        # Guarantess that at least end or start is in interval
        if middle_angle <0 :
            middle_angle += 2*np.pi
        # Extract image based on their middle angle
        middle_index =  int(np.rint((self.width)*(middle_angle)/(2*np.pi)))
        start_index = middle_index - self.width_pixel //2
        end_index = middle_index + self.width_pixel //2
        # Extract snippet
        if start_index >=0 and end_index < self.width:
            x = x[:, start_index:end_index]
            y = y[:, start_index:end_index]
        # Boundary case
        elif end_index >= self.width:
            x = np.concatenate([x[:,start_index:],x[:,:end_index-self.width]], axis = 1)
            y = np.concatenate([y[:,start_index:],y[:,:end_index-self.width]], axis = 1)
        elif start_index <0:
            x = np.concatenate([x[:, start_index+self.width:], x[:, :end_index]], axis = 1)
            y = np.concatenate([y[:, start_index+self.width:], y[:, :end_index]], axis = 1)

        # horizontal_flip with probability
        # horizontal_flip with probability 0.5
        flip_prob = np.random.uniform(0.0, 1.0)
        if flip_prob > self.flip_probability:
            x, y = x[:,::-1,:], y[:,::-1,:]

        #channel first
        x = x.transpose(2,0,1)
        y = y.transpose(2,0,1)

        x, y = torch.tensor(x.copy()), torch.tensor(y.copy(), requires_grad=False)
        return x, y


if __name__ == '__main__':

    root_dir = './dataset_sample/'

    data_path= glob.glob(root_dir+'/*.npy')

    trans_train = CloudStorTransformer()

    transform_list = trans_train

    dset = CloudStorDataset(data_path, transform=transform_list)

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