#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/15 9:57
# @Author  : Ws
# @File    : net.py
# @Software: PyCharm
import os

import lightning as L
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch import nn
from matplotlib import pyplot as plt

from model.unet import UNet


class Net(L.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.net = UNet(
            in_channels=1,
            model_channels=64,
            out_channels=1,
            num_res_blocks=2,
            num_heads=4,
            num_heads_upsample=-1,
            num_head_channels=32,
            attention_resolutions=(16, 8, 4)
        )
        self.compute_loss = nn.MSELoss()
        self.lr = lr

    def forward(self, x, height):
        return self.net(x, height)

    def training_step(self, batch, batch_idx):
        out = self.forward(batch['model'], batch['height'])
        loss = self.compute_loss(out, batch['data'])

        self.log_dict({'train_loss': loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch['model'], batch['height'])
        loss = self.compute_loss(out, batch['data'])
        plt.figure(figsize=(10, 6))
        plt.plot(out.cpu().numpy()[0][0][0,0,:], label='Pre')
        plt.plot(batch['data'].cpu().numpy()[0][0][0,0,:], label='True')
        plt.savefig('./log_release/log')
        plt.close()

        self.log_dict(
            {'val_loss': loss,
             'psnr': psnr(out[0][0].cpu().numpy(), batch['data'][0][0].cpu().numpy(), data_range=1),
             'ssim': ssim(out[0][0].cpu().numpy(), batch['data'][0][0].cpu().numpy(), data_range=1),
             }, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9,
                                                                       0.95))
