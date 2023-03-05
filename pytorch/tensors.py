import torch
from torchvision import datasets

def run():
    cifar=datasets.CIFAR10('./cifar/')
    data=torch.tensor(cifar.data)
    print(data.size())

if __name__=='__main__':
    run()