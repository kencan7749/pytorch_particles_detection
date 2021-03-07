import os 
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from losses import DiceLoss, F1_Loss
import pytorch_lightning as pl
       

def conv_block(in_channels, out_channels):
    """convolutional block
    conv-bn-relu-conv-bn-relu

    Args:
        in_channels ([int]): input channel
        out_channels ([int]): output channel

    Returns:
        [type]: [description]
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (3,3), padding=1, padding_mode='zeros'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, (3,3), padding=1, padding_mode='zeros'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def encoder_block(in_channels, out_channels):
    encoder = conv_block(in_channels, out_channels)
    encoder_pool = nn.MaxPool2d((1,2), stride=(1,2))

    return encoder

def decoder_block(concat_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(concat_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(concat_channels, out_channels, (3,3), padding=1, padding_mode='zeros'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, (3,3), padding=1, padding_mode='zeros'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class LitUNet(pl.LightningModule):

    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.conv1 = conv_block(in_channels, 16)
        self.conv2 = conv_block(16,32)
        self.conv3 = conv_block(32, 64)
        self.conv4 = conv_block(64,128)
        self.conv5 =  nn.Conv2d(128, 256 , (3,3), padding=1, padding_mode='zeros')

        self.deconv4 = nn.ConvTranspose2d(256, 128, (1,2), stride=(1,2))
        self.sconv4 = decoder_block(128+128, 128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, (1,2), stride=(1,2))
        self.sconv3 = decoder_block(64 + 64, 64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, (1,2), stride=(1,2))
        self.sconv2 = decoder_block(32 + 32, 32)
        self.deconv1 = nn.ConvTranspose2d(32, 16, (1,2), stride=(1,2))
        self.sconv1 = decoder_block(16+16, 16)
        self.sconv0 = nn.Conv2d(16, out_channels, (1,1))


        self.dice_loss = DiceLoss()
        self.f1_loss = F1_Loss()

    def forward(self, x):
        enc1 = self.conv1(x)
        #max pool
        x = F.max_pool2d(enc1,(1,2), stride=(1,2)) #causion channel
        enc2 = self.conv2(x)
        #max pool
        x = F.max_pool2d(enc2,(1,2), stride=(1,2)) #causion channel
        enc3 = self.conv3(x)
        #max pool
        x = F.max_pool2d(enc3,(1,2), stride=(1,2)) #causion channel
        enc4 = self.conv4(x)
        #max pool
        x = F.max_pool2d(enc4,(1,2), stride=(1,2)) #causion channel
        x = self.conv5(x)
        # expansiton
        x = self.deconv4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.sconv4(x)
        
        x = self.deconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.sconv3(x)

        x = self.deconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.sconv2(x)

        x = self.deconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.sconv1(x)

        x = self.sconv0(x)

        return x 

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.forward(x)
        y_pred = F.softmax(y_out)

        #Calculate loss
        bce_loss = F.binary_cross_entropy(y_pred, y)
        dice_loss = self.dice_loss(y_pred, y)
        loss = bce_loss + dice_loss

        self.log('training_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('bce_loss', bce_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('dice_loss', dice_loss,prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.shape[0]
        y_out = self.forward(x)
        y_pred = F.softmax(y_out, dim=1)

        #Calculate loss
        bce_loss = F.binary_cross_entropy(y_pred, y)
        dice_loss = self.dice_loss(y_pred, y)
        loss = bce_loss + dice_loss

        y_pred_label =torch.argmax(y,1)
        y_label = torch.argmax(y,1 )
        f1_loss = self.f1_loss(y_pred_label.view(batch_size, -1), y_label.view(batch_size, -1))

        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val_bce_loss', bce_loss,prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log('val_dice_loss', dice_loss,prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log('val_f1_loss', f1_loss,prog_bar=True, logger=True, on_step=True, on_epoch=True)

    def validation_end(self, outputs):
         # Optional
         avg_loss = torch.stack(x['val_loss'] for x in outputs).mean()
         tensorboard_logs = {'val_loss': avg_loss}
         return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
        



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer





if __name__ == '__main__':
    print('start')

    inp = torch.rand(1,4,32,512)

    model = LitUNet()

    out= model.forward(inp)


    print('done')
