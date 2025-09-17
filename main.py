#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/15 21:38
# @Author  : Ws
# @File    : main.py
# @Software: PyCharm
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner import Tuner
from lightning import Trainer

from data_modul import DataModule
from net import Net


def train():
    data = DataModule(train_dir='./dataset/release_train', val_dir='./dataset/release_train', batch_size=1)
    net = Net()
    # accumulator = GradientAccumulationScheduler(scheduling={0: 8, 4: 4, 8: 1})
    # TODO 更改Checkpoint保存策略
    checkpoint_callback = ModelCheckpoint(dirpath="./log_release",
                                          save_top_k=2,
                                          monitor="val_loss",
                                          save_last=False,
                                          every_n_epochs=50,
                                          save_on_train_epoch_end=True,
                                          filename="{epoch:04d}")

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
        # precision="16-mixed",
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor],
        accumulate_grad_batches=2,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        max_epochs=2000,
        default_root_dir="./log_release",
        check_val_every_n_epoch=10)

    trainer.fit(model=net, datamodule=data)


if __name__ == '__main__':
    train()
