import time
import keyboard
import torch
from torchvision.models import vgg16_bn


m = vgg16_bn(num_classes=8)
print(m)
t = torch.rand((1, 3, 210, 180))
tt = m(t)
print(tt.shape)
# for i in range(400):
#     k = keyboard.read_key()
#     print(k)
#     time.sleep(0.3)



# Dataset
# Dataloader
# Optimizer