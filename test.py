import os

import numpy as np
import torch
from net import Net

model_data = np.load('test/1.npy')
model_data = np.log(model_data)
model_data = model_data / 3
model_data = model_data[np.newaxis, :][np.newaxis, :].astype(np.float32)
model_data = torch.tensor(model_data).cuda()

height = torch.tensor([30], dtype=torch.float32).cuda()

model = Net.load_from_checkpoint('log/epoch=199.ckpt')
model.eval()
with torch.no_grad():
    pred = model(model_data, height)

print(pred.shape)
