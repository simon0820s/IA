import torch
import torch.nn as nn
import torch.nn.functional as F

def run():
    class Net(nn.Module):
        def __init__(self,num_channels):
            super(Net,self).__init__()

            self.num_channels=num_channels
            self.conv1=nn.Conv2d(3,self.num_channels,kernel_size=3,stride=1,padding=1)
            self.conv2=nn.Conv2d(self.num_channels,self.num_channels*2,kernel_size=3,stride=1,padding=1)
            self.conv3=nn.Conv2d(self.num_channels*2,self.num_channels*4,kernel_size=3,stride=1,padding=1)
            self.fc1=nn.Linear(self.num_channels*4*8*8,self.num_channels*4)
            self.fc2=nn.Linear(self.num_channels*4,6)

            def forward(self,x):

                x=self.conv1(x)
                x=F.relu(F.max_pool2d(x,2))

                x=self.conv2(x)
                x=F.relu(F.max_pool2d(x,2))

                x=self.conv3(x)
                x=F.relu(F.max_pool2d(x,2))

                x=x.view(-1,self.num_channels*4*8*8)

                x=self.fc1(x)
                x=F.relu(x)
                x=self.fc2(x)
                x=F.log_softmax(x,dim=1)

                return x

if __name__=='__main__':
    run()