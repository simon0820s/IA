import torch
import torch.nn as nn


def run():
    linear=nn.Linear(in_features=4096,out_features=10)
    conv=nn.Conv2d(in_channels=3,out_channels=9,kernel_size=3,stride=1,padding=1)
    relu=nn.ReLU(False)

if __name__=='__main__':
    run()