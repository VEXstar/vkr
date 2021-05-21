import torch.nn as nn
import torch.nn as torch__nn
import torch.nn.functional as F
import torch
import os
from django.conf import settings
from vkr.loader.ct_loader import SIZE
from tqdm import tqdm

class UNet(torch__nn.Module):
    def __init__(self):
        super().__init__()

        prob = 0.6
        self.drop = nn.Dropout(prob)

        # encoder (downsampling)
        self.enc_conv_0_1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.batch_norm_0_1 = nn.BatchNorm2d(64)
        self.enc_conv_0_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.batch_norm_0_2 = nn.BatchNorm2d(64)
        self.pool_0 = nn.MaxPool2d(2, 2)  # 256 -> 128

        self.enc_conv_1_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.batch_norm_1_1 = nn.BatchNorm2d(128)
        self.enc_conv_1_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.batch_norm_1_2 = nn.BatchNorm2d(128)
        self.pool_1 =  nn.MaxPool2d(2,2) # 128 -> 64

        self.enc_conv_2_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.batch_norm_2_1 = nn.BatchNorm2d(256)
        self.enc_conv_2_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.batch_norm_2_2 = nn.BatchNorm2d(256)
        self.pool_2 = nn.MaxPool2d(2, 2)  # 64 -> 32

        self.enc_conv_3_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.batch_norm_3_1 = nn.BatchNorm2d(512)
        self.enc_conv_3_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.batch_norm_3_2 = nn.BatchNorm2d(512)
        self.pool_3 =  nn.MaxPool2d(2, 2) # 32 -> 16

        # bottleneck
        self.bottleneck_conv_1 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.bottleneck_batch_norm_1 = nn.BatchNorm2d(1024)
        self.bottleneck_conv_2 = nn.Conv2d(1024, 512, 3, 1, 1)
        self.bottleneck_batch_norm_2 = nn.BatchNorm2d(512)

        # decoder (upsampling)
        self.upsample_0 = nn.Upsample(scale_factor=2, mode='nearest') # 16 -> 32
        self.dec_conv_0_1 =  nn.Conv2d(1024,512,3,1,1)
        self.batch_norm_4_1 = nn.BatchNorm2d(512)
        self.dec_conv_0_2 = nn.Conv2d(512, 256, 3, 1, 1)
        self.batch_norm_4_2 = nn.BatchNorm2d(256)

        self.upsample_1 = nn.Upsample(scale_factor=2, mode='nearest') # 32 -> 64
        self.dec_conv_1_1 = nn.Conv2d(512,256,3,1,1)
        self.batch_norm_5_1 = nn.BatchNorm2d(256)
        self.dec_conv_1_2 = nn.Conv2d(256, 128, 3, 1, 1)
        self.batch_norm_5_2 = nn.BatchNorm2d(128)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='nearest') # 64 -> 128
        self.dec_conv_2_1 = nn.Conv2d(256,128,3,1,1)
        self.batch_norm_6_1 = nn.BatchNorm2d(128)
        self.dec_conv_2_2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.batch_norm_6_2 = nn.BatchNorm2d(64)

        self.upsample_3 =  nn.Upsample(scale_factor=2, mode='nearest') # 128 -> 256
        self.dec_conv_3_1 = nn.Conv2d(128,64,3,1,1)
        self.batch_norm_7_1 = nn.BatchNorm2d(64)
        self.dec_conv_3_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.batch_norm_7_2 = nn.BatchNorm2d(64)
        self.dec_conv_3_3 = nn.Conv2d(64, 1, 3, 1, 1) # по схеме выход должен быть равен 2 + 1*1, а не 3*3

    def forward(self, x):
        # encoder
        e0 = (F.relu(self.batch_norm_0_2(self.enc_conv_0_2(F.relu(self.batch_norm_0_1(self.enc_conv_0_1(x)))))))
        e0_p = self.drop(self.pool_0(e0))
        e1 = (F.relu(self.batch_norm_1_2(self.enc_conv_1_2(F.relu(self.batch_norm_1_1(self.enc_conv_1_1(e0_p)))))))
        e1_p = self.drop(self.pool_1(e1))
        e2 = (F.relu(self.batch_norm_2_2(self.enc_conv_2_2(F.relu(self.batch_norm_2_1(self.enc_conv_2_1(e1_p)))))))
        e2_p = self.drop(self.pool_2(e2))
        e3 = (F.relu(self.batch_norm_3_2(self.enc_conv_3_2(F.relu(self.batch_norm_3_1(self.enc_conv_3_1(e2_p)))))))
        e3_p = self.drop(self.pool_3(e3))

        # bottleneck
        b =  F.relu(self.bottleneck_batch_norm_2(self.bottleneck_conv_2(
            F.relu(self.bottleneck_batch_norm_1(self.bottleneck_conv_1(e3_p))))))

        # decoder
        d0 = F.relu(self.batch_norm_4_2(self.dec_conv_0_2(F.relu(
            self.batch_norm_4_1(self.dec_conv_0_1(torch.cat([self.upsample_0(b), e3],1)))))))
        d1 = F.relu(self.batch_norm_5_2(self.dec_conv_1_2(F.relu(
            self.batch_norm_5_1(self.dec_conv_1_1(torch.cat([self.upsample_1(d0), e2],1)))))))
        d2 = F.relu(self.batch_norm_6_2(self.dec_conv_2_2(F.relu(
            self.batch_norm_6_1(self.dec_conv_2_1(torch.cat([self.upsample_2(d1), e1],1)))))))
        d3 = F.sigmoid(self.dec_conv_3_3(F.relu(self.batch_norm_7_2(self.dec_conv_3_2(F.relu(
            self.batch_norm_7_1(self.dec_conv_3_1(torch.cat([self.upsample_3(d2), e0],1)))))))))
        return d3


mode = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet()
model.load_state_dict(torch.load(os.path.join(settings.BASE_DIR, 'content/prod_model_'+str(SIZE)+'.pt'), map_location=mode))
model.to(mode)
model.eval()


def predict(data):
    predicted_mask = []
    t = tqdm(data, desc='Phase one')
    if not torch.cuda.is_available():
        print("GPU not found, using CPU")
    for (norm, real) in t:

        pred_batch = model(norm.to(mode)) > 0.5
        for mask in pred_batch:
            predicted_mask.append(mask.cpu().numpy())
    torch.cuda.empty_cache()
    return predicted_mask
