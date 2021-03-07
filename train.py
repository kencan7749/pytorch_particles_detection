import os
import glob
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import loggers

from model import LitUNet
from dataloaders import CloudStorTransformer, CloudStorDataset

root_dir = './dataset_sample/'
file_names = ["1-dust", "2-dust", "3-dust", "4-dust", "5-dust", "6-dust", "7-dust",
                  "8-dust", "9-smoke", "10-smoke", "11-smoke", "12-smoke", "13-smoke",
                  "14-smoke", "15-smoke", "16-smoke", "17-smoke", "18-smoke", "19-smoke"]
metapath = "metadata.npy"
batch_size = 32
feat = "dual"


train_indices = [1, 2, 3, 6, 8, 10, 11, 13, 16]
test_indices = [7,18]
train_indices = [1, 2]
test_indices = [1,2]

train_data_path= sum([glob.glob(root_dir+ f"/{i}*.npy") for i in train_indices], [])
test_data_path= sum([glob.glob(root_dir+ f"/{i}*.npy") for i in test_indices], [])

trans_train = CloudStorTransformer()
trans_valid = CloudStorTransformer(probability=0.0)

train_data_set = CloudStorDataset(train_data_path, feat=feat, transform=trans_train)
valid_data_set = CloudStorDataset(test_data_path, feat=feat, transform=trans_valid)

train_data_loader = torch.utils.data.DataLoader(
    train_data_set, batch_size=batch_size, shuffle=True
)
valid_data_loader = torch.utils.data.DataLoader(
    valid_data_set, batch_size=batch_size, shuffle=False
)

print(torch.cuda.is_available())
model = LitUNet()
#Gpu利用のためにdevice定義
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#GPUに載せる
#model = model.to(device)

logger = loggers.TensorBoardLogger('logs/')

checkpointing = pl.callbacks.ModelCheckpoint(monitor='val_loss')
trainer = pl.Trainer(gpus=1,logger=logger)

trainer.fit(model, train_data_loader, valid_data_loader)

