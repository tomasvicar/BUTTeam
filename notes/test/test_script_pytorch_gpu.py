import torch
import time

a=torch.zeros(5)

a=a.cuda()

print(a)

time.sleep(20)
